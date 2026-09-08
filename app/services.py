from __future__ import annotations

from functools import lru_cache

from app.config import get_settings
from app.database.qdrant import get_vector_store
from app.database.redis import get_cache
from app.generation.answer import AnswerGenerator
from app.generation.citations import build_citations
from app.generation.notes import NotesGenerator
from app.generation.quiz import QuizGenerator
from app.generation.summary import SummaryGenerator
from app.ingestion.chunker import SemanticTimestampChunker
from app.ingestion.metadata import VideoRepository
from app.ingestion.transcript import fetch_transcript
from app.ingestion.youtube import fetch_metadata
from app.llm.groq_client import GroqLLM
from app.observability.metrics import QueryMetrics
from app.observability.tracing import Trace
from app.reasoning.evidence_grader import EvidenceGrader
from app.reasoning.grounding_checker import GroundingChecker
from app.reasoning.query_decomposer import merge_query_text
from app.reasoning.query_rewriter import choose_search_queries
from app.reasoning.router import QueryRouter
from app.retrieval.embeddings import get_dense_encoder
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.reranker import get_reranker
from app.schemas.chat import ChatRequest, ChatResponse, ContentResponse, QuizResponse
from app.schemas.retrieval import RetrievedChunk
from app.schemas.video import VideoInfo


class VideoService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.repo = VideoRepository()
        self.encoder = get_dense_encoder()
        self.vector_store = get_vector_store()
        self.chunker = SemanticTimestampChunker()

    def process(self, url: str) -> VideoInfo:
        trace = Trace("process_video")
        with trace.span("metadata"):
            metadata = fetch_metadata(url)
            trace.video_id = metadata.video_id
        with trace.span("transcript"):
            segments = fetch_transcript(metadata.video_id)
        with trace.span("semantic_chunking"):
            chunks = self.chunker.chunk(
                video_id=metadata.video_id,
                title=metadata.title,
                segments=segments,
                encode_texts=self.encoder.encode_documents,
            )
        if not chunks:
            raise RuntimeError("The transcript could not be converted into searchable sections.")
        with trace.span("embedding"):
            vectors = self.encoder.encode_documents([chunk.text for chunk in chunks])
        with trace.span("qdrant_index"):
            self.vector_store.index(metadata.video_id, chunks, vectors, self.encoder.dimension)
        duration = max((segment.start + segment.duration for segment in segments), default=0.0)
        video = VideoInfo(
            video_id=metadata.video_id,
            title=metadata.title,
            url=metadata.url,
            thumbnail_url=metadata.thumbnail_url,
            chunk_count=len(chunks),
            duration_seconds=duration,
        )
        self.repo.save(video, chunks)
        from app.retrieval.sparse import get_sparse_registry

        get_sparse_registry().invalidate(metadata.video_id)
        trace.finish(chunk_count=len(chunks), duration_seconds=duration)
        return video


class RAGService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.repo = VideoRepository()
        self.retriever = HybridRetriever()

    def _llm_components(self):
        llm = GroqLLM()
        return (
            QueryRouter(llm),
            EvidenceGrader(llm),
            AnswerGenerator(llm),
            GroundingChecker(llm),
            SummaryGenerator(llm),
            NotesGenerator(llm),
            QuizGenerator(llm),
        )

    def _retrieve_plan(self, video_id: str, plan) -> list[RetrievedChunk]:
        queries = choose_search_queries(plan.standalone_query, plan.subqueries if plan.needs_decomposition else [])
        ranked_lists: list[list[RetrievedChunk]] = []
        for query in queries:
            ranked_lists.append(
                self.retriever.retrieve(
                    video_id=video_id,
                    query=query,
                    start_time=plan.start_time if plan.use_timestamp_filter else None,
                    end_time=plan.end_time if plan.use_timestamp_filter else None,
                )
            )
        if len(ranked_lists) == 1:
            return ranked_lists[0]
        fused = reciprocal_rank_fusion(ranked_lists, limit=18)
        rerank_query = merge_query_text(plan.standalone_query, queries)
        return get_reranker().rerank(rerank_query, fused, self.settings.rerank_top_k)

    def chat(self, request: ChatRequest) -> ChatResponse:
        self.repo.get_video(request.video_id)
        trace = Trace("chat", request.video_id)
        metrics = QueryMetrics()
        router, grader, answerer, grounding, summaryer, noteser, quizzer = self._llm_components()

        with trace.span("query_plan"):
            plan = router.plan(request.question, request.history)
            metrics.route = plan.route
            trace.event(
                "query_plan_result",
                route=plan.route,
                standalone_query=plan.standalone_query,
                subqueries=plan.subqueries,
                timestamp_filter=[plan.start_time, plan.end_time] if plan.use_timestamp_filter else None,
            )

        # Chat can understand these intents too, even though the UI has dedicated modes.
        if plan.route == "SUMMARY":
            result = summaryer.generate(self.repo.get_chunks(request.video_id))
            trace.finish(route=plan.route)
            return ChatResponse(answer=result.content, sources=result.sources)
        if plan.route == "NOTES":
            result = noteser.generate(self.repo.get_chunks(request.video_id))
            trace.finish(route=plan.route)
            return ChatResponse(answer=result.content, sources=result.sources)
        if plan.route == "QUIZ":
            trace.finish(route=plan.route, redirected=True)
            return ChatResponse(
                answer="Use the Quiz tab to start an interactive quiz from this video.",
                sources=[],
            )

        with trace.span("hybrid_retrieval"):
            evidence = self._retrieve_plan(request.video_id, plan)
            metrics.retrieved_chunk_ids = [chunk.chunk_id for chunk in evidence]
            trace.event(
                "retrieval_result",
                results=[
                    {
                        "chunk_id": chunk.chunk_id,
                        "dense_score": chunk.dense_score,
                        "sparse_score": chunk.sparse_score,
                        "fusion_score": chunk.fusion_score,
                        "rerank_score": chunk.rerank_score,
                    }
                    for chunk in evidence
                ],
            )

        with trace.span("evidence_grade"):
            grade = grader.grade(plan.standalone_query, evidence)
            metrics.evidence_grade = grade.label
            trace.event("evidence_grade_result", label=grade.label, improved_query=grade.improved_query)

        if grade.label == "PARTIALLY_RELEVANT" and self.settings.max_retrieval_retries > 0:
            metrics.retrieval_retries += 1
            with trace.span("corrective_retrieval"):
                corrected = self.retriever.retrieve(
                    video_id=request.video_id,
                    query=grade.improved_query,
                    start_time=plan.start_time if plan.use_timestamp_filter else None,
                    end_time=plan.end_time if plan.use_timestamp_filter else None,
                )
                combined = reciprocal_rank_fusion([evidence, corrected], limit=16)
                evidence = get_reranker().rerank(
                    plan.standalone_query,
                    combined,
                    self.settings.rerank_top_k,
                )
            with trace.span("corrected_evidence_grade"):
                grade = grader.grade(plan.standalone_query, evidence)
                metrics.evidence_grade = grade.label

        if grade.label == "IRRELEVANT" or not evidence:
            trace.finish(
                route=metrics.route,
                evidence_grade=metrics.evidence_grade,
                retrieval_retries=metrics.retrieval_retries,
                insufficient_evidence=True,
            )
            return ChatResponse(
                answer="I couldn't find enough information in this video to answer that reliably.",
                sources=[],
                insufficient_evidence=True,
            )

        with trace.span("answer_generation"):
            answer = answerer.generate(request.question, evidence, request.history)

        with trace.span("grounding_check"):
            check = grounding.check(request.question, answer, evidence)
            metrics.grounded = check.grounded

        if not check.grounded and self.settings.max_generation_retries > 0:
            metrics.generation_retries += 1
            instruction = "Remove or correct these unsupported claims: " + "; ".join(check.unsupported_claims[:6])
            with trace.span("answer_regeneration"):
                answer = answerer.generate(request.question, evidence, request.history, instruction)
            with trace.span("regenerated_grounding_check"):
                check = grounding.check(request.question, answer, evidence)
                metrics.grounded = check.grounded

        if not check.grounded:
            answer = "I found related parts of the video, but I couldn't verify a reliable answer from them."

        citations = build_citations(evidence, 4) if check.grounded else []
        trace.finish(
            route=metrics.route,
            evidence_grade=metrics.evidence_grade,
            grounded=metrics.grounded,
            retrieval_retries=metrics.retrieval_retries,
            generation_retries=metrics.generation_retries,
            retrieved_chunk_ids=metrics.retrieved_chunk_ids,
        )
        return ChatResponse(
            answer=answer,
            sources=citations,
            insufficient_evidence=not check.grounded,
        )

    def summary(self, video_id: str) -> ContentResponse:
        self.repo.get_video(video_id)
        cache = get_cache()
        key = f"summary:{video_id}"
        cached = cache.get_json(key)
        if cached:
            return ContentResponse.model_validate(cached)
        llm = GroqLLM()
        result = SummaryGenerator(llm).generate(self.repo.get_chunks(video_id))
        cache.set_json(key, result.model_dump(), 3600)
        return result

    def notes(self, video_id: str) -> ContentResponse:
        self.repo.get_video(video_id)
        cache = get_cache()
        key = f"notes:{video_id}"
        cached = cache.get_json(key)
        if cached:
            return ContentResponse.model_validate(cached)
        llm = GroqLLM()
        result = NotesGenerator(llm).generate(self.repo.get_chunks(video_id))
        cache.set_json(key, result.model_dump(), 3600)
        return result

    def quiz(self, video_id: str) -> QuizResponse:
        self.repo.get_video(video_id)
        cache = get_cache()
        key = f"quiz:{video_id}"
        cached = cache.get_json(key)
        if cached:
            return QuizResponse.model_validate(cached)
        llm = GroqLLM()
        result = QuizGenerator(llm).generate(self.repo.get_chunks(video_id))
        cache.set_json(key, result.model_dump(), 3600)
        return result


@lru_cache(maxsize=1)
def get_video_service() -> VideoService:
    return VideoService()


@lru_cache(maxsize=1)
def get_rag_service() -> RAGService:
    return RAGService()
