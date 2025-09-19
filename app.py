# app.py
import base64
import json
import os
from io import BytesIO

import requests
import streamlit as st
from PIL import Image

# Carrega variáveis de ambiente a partir de um .env (se existir)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ====== CONFIG FIXA DO SEU PROJETO (edite aqui) ======
PROJECT_ID   = "527566758229"
LOCATION     = "us-central1"        # ajuste se necessário
ENDPOINT_ID  = "4481815003489370112"  # confira se está correto (antes você mostrou um com '5803...')
CONFIDENCE_THRESHOLD = 0.5
MAX_PREDICTIONS     = 5
# =====================================================

# --- Auth ADC (Application Default Credentials) ---
# Funciona com:
#   gcloud auth application-default login
# ou variável GOOGLE_APPLICATION_CREDENTIALS apontando para um key.json de service account
from google.auth.transport.requests import Request
import google.auth

SCOPE = ["https://www.googleapis.com/auth/cloud-platform"]

def get_access_token_adc():
    """Obtém um access token usando Application Default Credentials (ADC)."""
    creds, _ = google.auth.default(scopes=SCOPE)
    if not creds.valid:
        creds.refresh(Request())
    return creds.token

def image_to_b64(file_bytes: bytes) -> str:
    return base64.b64encode(file_bytes).decode("utf-8")

def predict_request(image_b64: str, token: str):
    """Chama o endpoint do Vertex AI com a imagem em base64."""
    url = (
        f"https://{LOCATION}-aiplatform.googleapis.com/v1/"
        f"projects/{PROJECT_ID}/locations/{LOCATION}/endpoints/{ENDPOINT_ID}:predict"
    )
    payload = {
        "instances": [{"content": image_b64}],
        "parameters": {
            "confidenceThreshold": CONFIDENCE_THRESHOLD,
            "maxPredictions": MAX_PREDICTIONS,
        },
    }
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    return requests.post(url, headers=headers, data=json.dumps(payload))

def interpretar_resposta(data: dict) -> dict:
    """
    Lê a resposta do Vertex AI de forma robusta.
    Aceita variações de chaves e corrige tipos (ex.: número -> [número]).
    Retorna: classe (str), confianca (float), mensagem (str), alerta (str)
    """
    import json

    def _as_dict(obj):
        # Se vier string JSON, tenta fazer parse
        if isinstance(obj, str):
            try:
                return json.loads(obj)
            except Exception:
                return {}
        return obj if isinstance(obj, dict) else {}

    def _first_from(obj, keys):
        """Tenta pegar o primeiro valor existente entre várias chaves possíveis."""
        for k in keys:
            if isinstance(obj, dict) and k in obj:
                return obj[k]
        return None

    def _as_list(x):
        if x is None:
            return []
        if isinstance(x, list):
            return x
        return [x]

    try:
        data = _as_dict(data)
        preds = data.get("predictions") or data.get("preds") or []
        if not isinstance(preds, list) or not preds:
            raise ValueError("Campo 'predictions' ausente ou vazio.")

        p0 = preds[0] if isinstance(preds[0], dict) else {}
        # Variações comuns
        display_names = _first_from(p0, ["displayNames", "displaynames", "display_names", "classes", "labels"])
        confidences   = _first_from(p0, ["confidences", "confidence", "scores", "score", "probabilities", "probability"])

        display_names = _as_list(display_names)
        confidences   = _as_list(confidences)

        if not display_names or not confidences:
            raise ValueError("Não encontrei 'displayNames'/'confidences' (ou equivalentes).")

        classe = str(display_names[0])

        # garante float mesmo que venha string/objeto
        try:
            confianca = float(confidences[0])
        except Exception:
            if isinstance(confidences[0], dict):
                num = next(
                    (v for v in confidences[0].values() if isinstance(v, (int, float, str))),
                    0.0
                )
                confianca = float(num)
            else:
                confianca = 0.0



        if classe.lower() == "nondemented":
            mensagem = (
                "🟢 **Classificação:** *Não demente* (nonDemented).\n\n"
                "A imagem **não apresenta padrões típicos** de Alzheimer"
            )
        else:
            mensagem = (
                "🔴 **Classificação:** *Demente* (Demented).\n\n"
                "A imagem **apresenta padrões compatíveis** com doença de Alzheimer"

            )

        return {"classe": classe, "confianca": confianca, "mensagem": mensagem}

    except Exception as e:
        # fallback
        return {
            "classe": "Desconhecida",
            "confianca": 0.0,
            "mensagem": f"Não foi possível interpretar a resposta do modelo. Detalhe: {e}",
            "alerta": "Este resultado não constitui diagnóstico médico.",
        }

# --- Integração com OpenAI (GPT) para explicar baseado NA IMAGEM ---
from openai import OpenAI

def gpt_explain(image_bytes: bytes, status_label: str) -> str:
    """
    Gera explicação curta baseada na IMAGEM + rótulo/status ('Demented' | 'nonDemented').
    A imagem é a fonte primária de evidências; o status serve apenas de foco.
    Não usa o JSON além do status.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return ("[GPT] Falha: OPENAI_API_KEY não definida. "
                "Crie um .env com OPENAI_API_KEY=... ou exporte no ambiente.")

    client = OpenAI(api_key=api_key)

    # Envia a imagem como data URL (base64)
    data_url = f"data:image/jpeg;base64,{base64.b64encode(image_bytes).decode('utf-8')}"

    system_msg = (
        "Você é um assistente que descreve achados VISUAIS de tomografias de crânio "
        "em linguagem simples. Baseie-se primariamente no que é visível na imagem. "
        "Use o status (Demented/nonDemented) apenas como foco da análise. "
        "Não invente: se algo não estiver claro na imagem, diga que é incerto. "
        "Não forneça diagnóstico ou conduta clínica."
    )

    user_text = (
        f"Status do classificador: '{status_label}'. "
        f"Analise SOMENTE a imagem e explique, em 3–5 tópicos objetivos, "
        f"o que nela sustenta esse status. "
        f"Se 'Demented', foque em sinais compatíveis com Alzheimer (ex.: atrofia hipocampal, "
        f"aumento de ventrículos laterais, afinamento cortical temporoparietal). "
        f"Se 'nonDemented', foque na ausência desses padrões (volumes preservados, "
        f"proporções ventriculares esperadas, simetria). "
        f"Se não for possível identificar com segurança, diga que é incerto. "
        f"No final, inclua uma linha: 'Aviso: isto não é diagnóstico médico.' "
        f"Não mencione Vertex/JSON/tokens."
    )

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",  # pode usar "gpt-4o" para mais qualidade
            messages=[
                {"role": "system", "content": system_msg},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ],
            temperature=0.1,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        return f"[GPT] Erro ao gerar explicação: {e}"

# -------- UI --------
st.set_page_config(page_title="Classificação de Tomografia (Alzheimer)", page_icon="🧠", layout="centered")
st.title("🧠 Classificação de Tomografias")

# with st.expander("Informações fixas do projeto"):
#     st.write(f"**PROJECT_ID:** `{PROJECT_ID}`")
#     st.write(f"**LOCATION:** `{LOCATION}`")
#     st.write(f"**ENDPOINT_ID:** `{ENDPOINT_ID}`")

uploaded = st.file_uploader("Selecione uma imagem de tomografia", type=["jpg","jpeg","png","webp","bmp"])

if uploaded:
    try:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption=uploaded.name, use_column_width=True)
    except Exception as e:
        st.warning(f"Não consegui pré-visualizar: {e}")

send = st.button("🚀 Enviar para o modelo", type="primary", disabled=uploaded is None)

if send:
    try:
        # Lê bytes e codifica base64
        uploaded.seek(0)
        img_bytes = uploaded.read()
        b64 = image_to_b64(img_bytes)

        with st.status("Autenticando (ADC) e enviando ao Vertex...", expanded=False):
            token = get_access_token_adc()
            resp = predict_request(b64, token)

        st.subheader("Resposta")
        st.write(f"Status HTTP: {resp.status_code}")

        try:
            data = resp.json()
        except Exception:
            data = {"raw_text": resp.text}

        with st.expander("JSON bruto (debug)"):
            st.code(json.dumps(data, indent=2, ensure_ascii=False), language="json")

        if resp.ok:
            info = interpretar_resposta(data)

            # Mostra status e uma explicação curta
            st.markdown(
                f"**Classe prevista:** `{info['classe']}` | **Confiança:** `{info['confianca']:.3f}`"
            )
            st.info(info["mensagem"])

            # Explicação com GPT: IMAGEM é a evidência principal; status só guia
            st.subheader("Explicação automática com base na imagem")
            gpt_text = gpt_explain(img_bytes, info["classe"])
            st.write(gpt_text)
        else:
            st.error("A API retornou erro.")

    except Exception as e:
        st.error(
            "Falha ao autenticar/enviar. Verifique se você já executou "
            "`gcloud auth application-default login` ou configurou "
            "`GOOGLE_APPLICATION_CREDENTIALS` com um key.json válido."
        )
        st.exception(e)
