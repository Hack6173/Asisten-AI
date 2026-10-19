"""
asisten_ai.app_server - REST API Backend FastAPI untuk Asisten-AI Autonomous Intelligence
Menyediakan antarmuka terpadu untuk integrasi web, dashboard, dan kontrol eksekusi.
"""
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from asisten_ai.config import SYSTEM_CONFIG
from asisten_ai.resource_monitor import RESOURCE_MONITOR
from asisten_ai.offline_manager import OFFLINE_MANAGER
from asisten_ai.knowledge_store import KNOWLEDGE_STORE
from asisten_ai.retrieval_engine import RETRIEVAL_ENGINE
from asisten_ai.research_agent import RESEARCH_AGENT
from asisten_ai.learning_engine import LEARNING_ENGINE
from asisten_ai.dataset_builder import DATASET_BUILDER
from asisten_ai.training_manager import TRAINING_MANAGER
from asisten_ai.evaluation_engine import EVALUATION_ENGINE
from asisten_ai.model_registry import MODEL_REGISTRY
from asisten_ai.scheduler import SCHEDULER

app = FastAPI(
    title="Asisten-AI Autonomous Intelligence API",
    version="2.0.0",
    description="Sistem Kecerdasan Buatan Mandiri: Autonomous Research, Self-Learning, Self-Evaluation, and Continuous Model Training.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request & Response Schemas
class ChatRequest(BaseModel):
    message: str
    topic_hint: Optional[str] = None


class ResearchTriggerRequest(BaseModel):
    topic: Optional[str] = None
    questions: Optional[List[str]] = None


class PromoteModelRequest(BaseModel):
    candidate_id: str
    min_gain: float = 0.03
    force: bool = False


class RollbackModelRequest(BaseModel):
    target_model_id: str


class SchedulerControlRequest(BaseModel):
    action: str  # start, pause, resume, stop
    mode: Optional[str] = None


@app.get("/api/status")
def get_system_status():
    hw_stats = RESOURCE_MONITOR.get_realtime_stats()
    hw_cap = RESOURCE_MONITOR.assess_hardware_capabilities().to_dict()
    conn_status = OFFLINE_MANAGER.get_connectivity_status()
    active_model = MODEL_REGISTRY.get_active_model()
    sched_status = SCHEDULER.get_status().model_dump()
    summary = KNOWLEDGE_STORE.get_summary_statistics()

    return {
        "connectivity": conn_status,
        "hardware": hw_stats,
        "hardware_capability": hw_cap,
        "active_model": active_model.model_dump() if active_model else None,
        "scheduler": sched_status,
        "knowledge_summary": summary,
    }


@app.post("/api/chat")
def handle_chat(req: ChatRequest):
    # Cari fakta semantik dalam memori lokal
    retrieved = RETRIEVAL_ENGINE.search_semantic(req.message, topic_filter=req.topic_hint, top_k=3)
    response_data = RETRIEVAL_ENGINE.build_grounded_answer(req.message, retrieved)

    # Jika fakta tidak ditemukan atau kepercayaan 0, catat sebagai knowledge gap untuk diteliti otomatis
    if not response_data["has_grounding"]:
        OFFLINE_MANAGER.queue_pending_research(
            topic=req.topic_hint or req.message[:40],
            questions=[req.message],
            origin="user_chat_unanswered"
        )

    return {
        "answer": response_data["answer"],
        "has_grounding": response_data["has_grounding"],
        "citations": response_data["citations"],
        "confidence": response_data["confidence"],
        "retrieved_facts": [f.model_dump() for f in retrieved],
    }


@app.get("/api/research/gaps")
def get_knowledge_gaps():
    return RESEARCH_AGENT.identify_knowledge_gaps()


@app.post("/api/research/run")
def trigger_research(req: ResearchTriggerRequest):
    topic = req.topic or "Machine Learning Fundamentals"
    res = RESEARCH_AGENT.execute_autonomous_research(topic, target_questions=req.questions)
    return res.model_dump()


@app.post("/api/learning/run-cycle")
def run_learning_cycle(req: ResearchTriggerRequest):
    report = SCHEDULER.trigger_immediate_cycle(topic=req.topic)
    return report.model_dump()


@app.get("/api/knowledge")
def list_knowledge(topic: Optional[str] = None, limit: int = 50):
    return KNOWLEDGE_STORE.list_knowledge_items(topic=topic, limit=limit)


@app.delete("/api/knowledge/{item_id}")
def delete_knowledge(item_id: int):
    success = KNOWLEDGE_STORE.delete_knowledge_item(item_id)
    if not success:
        raise HTTPException(status_code=404, detail="Item pengetahuan tidak ditemukan.")
    return {"success": True}


@app.post("/api/dataset/build")
def build_dataset(name: str = "dataset_auto_v1"):
    res = DATASET_BUILDER.build_and_split_dataset(dataset_name=name)
    return res.model_dump()


@app.post("/api/training/run")
def train_candidate(dataset_name: str = "dataset_auto_v1", epochs: int = 3):
    train_file = str(DATASET_BUILDER.output_dir / f"{dataset_name}_train.jsonl")
    val_file = str(DATASET_BUILDER.output_dir / f"{dataset_name}_val.jsonl")

    # Pastikan dataset dibangun jika belum ada
    if not Path(train_file).exists():
        DATASET_BUILDER.build_and_split_dataset(dataset_name=dataset_name)

    result = TRAINING_MANAGER.run_training_experiment(
        dataset_name=dataset_name,
        train_file=train_file,
        val_file=val_file,
        epochs=epochs
    )

    if result.status == "completed" and result.checkpoint_dir:
        # Daftarkan otomatis sebagai kandidat
        MODEL_REGISTRY.register_candidate(
            candidate_name=f"Candidate-{dataset_name}",
            version=f"1.{int(result.duration_seconds)}.0",
            checkpoint_dir=result.checkpoint_dir,
            description=f"Model kandidat hasil pelatihan {dataset_name} ({result.total_steps} langkah)."
        )

    return result.model_dump()


@app.post("/api/training/interrupt")
def interrupt_training():
    TRAINING_MANAGER.interrupt_training()
    return {"message": "Sinyal penghentian pelatihan dikirim."}


@app.post("/api/evaluation/run")
def run_evaluation(model_name: Optional[str] = None):
    report = EVALUATION_ENGINE.run_evaluation(model_name=model_name or "Active-Asisten-Model")
    return report.model_dump()


@app.get("/api/evaluation/history")
def get_evaluation_history():
    return EVALUATION_ENGINE.get_evaluation_history()


@app.get("/api/models")
def get_models():
    return [m.model_dump() for m in MODEL_REGISTRY.list_models()]


@app.post("/api/models/promote")
def promote_candidate(req: PromoteModelRequest):
    success, msg, rep = MODEL_REGISTRY.evaluate_and_promote_candidate(
        candidate_id=req.candidate_id,
        min_gain=req.min_gain,
        force_promote=req.force
    )
    return {
        "success": success,
        "message": msg,
        "evaluation": rep.model_dump()
    }


@app.post("/api/models/rollback")
def rollback_model(req: RollbackModelRequest):
    success, msg = MODEL_REGISTRY.rollback_to_previous_model(req.target_model_id)
    return {"success": success, "message": msg}


@app.post("/api/scheduler/control")
def control_scheduler(req: SchedulerControlRequest):
    act = req.action.lower()
    if act == "start":
        SCHEDULER.start(mode=req.mode)
    elif act == "pause":
        SCHEDULER.pause()
    elif act == "resume":
        SCHEDULER.resume()
    elif act == "stop":
        SCHEDULER.stop()
    else:
        raise HTTPException(status_code=400, detail=f"Aksi scheduler tidak dikenal: {act}")

    return SCHEDULER.get_status().model_dump()


@app.get("/api/config")
def get_config():
    return SYSTEM_CONFIG.model_dump()


@app.post("/api/config")
def update_config(new_cfg: Dict[str, Any]):
    global SYSTEM_CONFIG
    try:
        updated = SYSTEM_CONFIG.model_validate(new_cfg)
        updated.save()
        SYSTEM_CONFIG = updated
        return {"success": True, "config": SYSTEM_CONFIG.model_dump()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Konfigurasi tidak valid: {e}")
