import os
import time
from pathlib import Path
from contextlib import asynccontextmanager
import uvicorn
import torch
from fastapi import FastAPI, HTTPException, APIRouter
from pydantic import BaseModel, Field
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline as hf_pipeline
from peft import PeftModel, PeftConfig
from typing import Dict, Any

# Configurações
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "resume_job_finetuned_model"

PROMPTS = {
    "generate_resume": (
        "Instruction: Gere um currículo profissional, claro, estruturado e totalmente otimizado, "
        "considerando todo o conteúdo fornecido pelo usuário. O currículo deve ser bem formatado, "
        "priorizando clareza, impacto profissional, medição de resultados e escrita de alta qualidade. "
        "Inclua seções como Resumo Profissional, Habilidades Técnicas, Experiências, Formação e Projetos "
        "quando aplicável. Não invente informações fora do Input.\n\n"
        "Input:\n{input}\n\n"
        "Response:"
    ),
    "generate_cover_letter": (
        "Instruction: Gere uma carta de apresentação personalizada, profissional e bem estruturada, "
        "orientada à vaga e alinhada às qualificações do candidato. Use linguagem formal e convincente, "
        "demonstre motivação, conecte as habilidades do candidato com as exigências da vaga, e termine com "
        "um parágrafo de encerramento forte. A carta deve parecer escrita por um humano experiente e "
        "persuasivo. Nunca invente fatos não mencionados.\n\n"
        "Input:\n{input}\n\n"
        "Response:"
    ),
    "simulate_interview": (
        "Instruction: Gere uma simulação completa de entrevista para a vaga desejada. "
        "Crie perguntas técnicas e comportamentais altamente relevantes e, para cada pergunta, gere uma "
        "resposta ideal, bem elaborada, realista e convincente. O tom deve ser profissional e adequado ao "
        "nível da vaga. As respostas devem demonstrar domínio técnico, clareza e segurança.\n\n"
        "Input:\n{input}\n\n"
        "Response:"
    ),
    "analyze_resume": (
        "Instruction: Analise profundamente o currículo do candidato em comparação com a descrição da vaga. "
        "Faça uma avaliação detalhada, identificando compatibilidades fortes, lacunas técnicas, possíveis riscos, "
        "pontos fracos e oportunidades de melhoria. Apresente também sugestões práticas e estratégicas para "
        "aumentar a compatibilidade com a vaga. Seja direto, analítico e preciso.\n\n"
        "Input:\n{input}\n\n"
        "Response:"
    ),
}


class GenRequest(BaseModel):
    task: str = Field(
        ...,
        description=(
            "Tarefa de geração desejada.\n"
            "Opções válidas:\n"
            "- generate_resume → Gera um resumo profissional curto\n"
            "- generate_cover_letter → Gera uma carta de apresentação\n"
            "- simulate_interview → Simula perguntas de entrevista\n"
            "- analyze_resume → Analisa e pontua um currículo\n"
        ),
        examples=["generate_resume"],
    )

    data: Dict[str, Any] = Field(
        ...,
        description=(
            "Dados de entrada necessários para cada tarefa.\n\n"
            "IMPORTANTE: O usuário deve inserir a experiência e habilidades apenas nos campos abaixo.\n\n"
            "Exemplos por tarefa:\n\n"
            "- generate_resume:\n"
            "  {\n"
            "    'experience': '[INSIRA A EXPERIÊNCIA AQUI]',\n"
            "    'skills': '[INSIRA AS HABILIDADES AQUI]'\n"
            "  }\n\n"
            "- generate_cover_letter:\n"
            "  {\n"
            "    'job_description': '[DESCRIÇÃO DA VAGA]',\n"
            "    'candidate_background': '[INFORMAÇÕES DO CANDIDATO]'\n"
            "  }\n\n"
            "- simulate_interview:\n"
            "  {\n"
            "    'job_title': '[CARGO ALVO]',\n"
            "    'stack': '[TECNOLOGIAS RELEVANTES]'\n"
            "  }\n\n"
            "- analyze_resume:\n"
            "  {\n"
            "    'resume': '[TEXTO DO CURRÍCULO]',\n"
            "    'job_description': '[REQUISITOS DA VAGA]'\n"
            "  }"
        ),
        examples=[
            {
                "experience": "[INSIRA A EXPERIÊNCIA DO USUÁRIO AQUI]",
                "skills": "[INSIRA AS HABILIDADES DO USUÁRIO AQUI]",
            }
        ],
    )

    max_tokens: int = Field(
        300,
        description="Quantidade máxima de tokens para gerar. Aumente para textos mais longos.",
        examples=[600],
    )

    num_beams: int = Field(
        3,
        description="Beam search. 3–5 recomendados para maior qualidade.",
        examples=[3],
    )


# Carregamento do Modelo
ai_models = {}


def load_model_pipeline(model_path: Path):
    print(f"🔄 Carregando modelo de: {model_path}")

    peft_config = PeftConfig.from_pretrained(str(model_path))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"Dispositivo detectado: {device.upper()} ({dtype})")

    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        peft_config.base_model_name_or_path,
        device_map="auto",
        torch_dtype=dtype,
    )

    model = PeftModel.from_pretrained(base_model, str(model_path))
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))

    return hf_pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
    )


# Lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        if MODEL_PATH.exists():
            ai_models["generator"] = load_model_pipeline(MODEL_PATH)
            print("✅ Modelo carregado com sucesso!")
        else:
            print(f"⚠️ Modelo não encontrado em {MODEL_PATH}")
    except Exception as e:
        print(f"❌ Erro ao carregar modelo: {e}")

    yield
    ai_models.clear()
    print("🛑 Aplicação encerrada.")


# FastAPI App
app = FastAPI(
    title="Skill Bridge Gen AI API",
    version="1.0.0",
    description="API de Inteligência Generativa para carreiras.",
    lifespan=lifespan,
    docs_url="/api/docs",
)

router = APIRouter(prefix="/api")


# Endpoint de Informações
@router.get("/", tags=["Info"])
def api_details():
    return {
        "app": "Skill Bridge Gen AI",
        "version": "1.0.0",
        "description": "API powered by LoRA Fine-tuned Model",
        "status": "online",
        "endpoints": {
            "generate": "/api/generate (POST)",
            "health": "/api/health (GET)",
            "docs": "/api/docs (GET)",
        },
        "tasks_available": list(PROMPTS.keys()),
    }


# Endpoint de Geração
@router.post("/generate", tags=["Geração"])
def generate(req: GenRequest):
    if req.task not in PROMPTS:
        raise HTTPException(status_code=400, detail="Task desconhecida")

    if "generator" not in ai_models:
        raise HTTPException(status_code=503, detail="Modelo não carregado.")

    # 🔥 Junta dados do usuário corretamente em multiline
    input_str = "\n".join([f"{k}: {v}" for k, v in req.data.items()])

    # 🔥 Monta o prompt seguindo o padrão do LoRA
    prompt = PROMPTS[req.task].format(input=input_str).strip()

    try:
        pipe = ai_models["generator"]
        start = time.time()

        # 🔥 Agora realmente permite respostas LONGAS
        out = pipe(
            prompt,
            max_new_tokens=req.max_tokens,
            num_beams=req.num_beams,
            do_sample=True,
            temperature=0.9,
            top_p=0.95,
            top_k=50,
            repetition_penalty=1.1,
            no_repeat_ngram_size=4,
            truncation=False,  # <--- NÃO TRUNCAR RESPOSTAS LONGAS
        )

        elapsed = time.time() - start

        # 🔥 Extrai corretamente o texto gerado
        generated_text = out[0].get("generated_text", "").strip()

        # 🔥 Remove o prompt do início (modelos seq2seq copiam o prompt)
        if "Response:" in generated_text:
            generated_text = generated_text.split("Response:", 1)[1].strip()

        return {
            "task": req.task,
            "input": req.data,
            "optimized_output": generated_text,
            "time_ms": round(elapsed * 1000, 3),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Health
@router.get("/health", tags=["Status"])
def health():
    return {"status": "ok", "model_loaded": "generator" in ai_models}


app.include_router(router)

# Executar
if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
