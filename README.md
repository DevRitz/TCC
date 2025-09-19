🧠 Alzheimer CT Classifier — Vertex AI + Streamlit + GPT

Aplicação em Python que envia uma tomografia para um Endpoint da Google Cloud Vertex AI, recebe a predição (Demented | nonDemented) e, em seguida, usa o GPT para gerar uma explicação curta baseada principalmente na imagem (o status só orienta o foco). Interface simples em Streamlit.

⚠️ Aviso: este software não produz diagnóstico médico. O resultado é apenas suporte computacional e deve ser interpretado por profissional habilitado.

✨ Recursos

Upload de imagem (JPG/PNG/WEBP/BMP) com preview

Conversão para Base64 e envio ao Vertex AI Endpoints

Autenticação automática via Application Default Credentials (ADC)

Interpretação de displayNames + confidences

Explicação automática com GPT baseada na própria imagem (status só guia)

UI minimalista focada no usuário final

🧱 Tecnologias

Python 3.x

Streamlit (UI)

Pillow (manipulação/preview da imagem)

requests (HTTP)

google-auth (token via ADC)

Google Cloud Vertex AI — Endpoints (inferência do modelo)

OpenAI (gpt-4o-mini / gpt-4o) (explicação visual)

python-dotenv (variáveis de ambiente)

🗂️ Estrutura
alzheimer-explainer/
├─ app.py
├─ requirements.txt
└─ .env.example


Sugestão de .gitignore:

__pycache__/
.venv/
.env
*.pyc
.DS_Store

🔧 Pré-requisitos

Python 3.9+

Google Cloud SDK instalado ou um arquivo de Service Account JSON

Chave da OpenAI (guarde com segurança)

⚙️ Configuração
1) Autenticação no Google (ADC)

No terminal, execute uma única vez por máquina:

gcloud auth application-default login


Alternativa: defina GOOGLE_APPLICATION_CREDENTIALS apontando para um JSON de Service Account com permissão no Vertex AI.

2) Variáveis de ambiente (OpenAI)

Copie .env.example para .env e adicione:

OPENAI_API_KEY=sua_chave_aqui

3) Dependências
pip install -r requirements.txt

4) IDs do Vertex

Edite no app.py:

PROJECT_ID   = "seu_project_id"
LOCATION     = "us-central1"   # ajuste se necessário
ENDPOINT_ID  = "seu_endpoint_id"

▶️ Executando
streamlit run app.py


Fluxo de uso:

Faça upload da tomografia

Clique em Enviar para o modelo

Veja:

Classe prevista (Demented | nonDemented) + confiança

Explicação curta do GPT baseada na imagem

🔎 Como funciona (resumo)

A UI lê a imagem e gera Base64

Obtém access token via ADC

Chama o Vertex AI Endpoint:
projects/{PROJECT_ID}/locations/{LOCATION}/endpoints/{ENDPOINT_ID}:predict

Interpreta displayNames[0] e confidences[0]

Envia a mesma imagem + status para o GPT → texto curto com achados visuais (sem diagnosticar)

🔐 Boas práticas & privacidade

Não diagnóstico: mantenha claro ao usuário final

Sigilo: trate PII e dados sensíveis com cuidado; evite logs contendo imagens/base64

Chaves: nunca versione OPENAI_API_KEY nem JSONs de credenciais

Custos: chamadas ao Vertex e ao GPT geram custo; monitore e faça rate limiting se necessário

🧯 Troubleshooting

gcloud: command not found → instale o Google Cloud SDK ou use GOOGLE_APPLICATION_CREDENTIALS

401/403 no Vertex → refaça gcloud auth application-default login ou revise permissões da Service Account

Streamlit não abre → verifique firewall/porta 8501

Token OpenAI ausente → defina OPENAI_API_KEY no .env ou ambiente

Endpoint ID incorreto → confirme o ENDPOINT_ID no console do Vertex AI

🗺️ Roadmap (ideias)

Upload de séries DICOM e slices

Auditoria de versões do modelo (Model Monitoring)

Logs estruturados e métricas (Prometheus/Grafana)

Opção de usar URL de imagem pública em vez de Base64

📜 Licença

Defina a licença do projeto (ex.: MIT).
Exemplo: MIT License — veja LICENSE.

🤝 Contribuições

Contribuições são bem-vindas!
Abra uma issue ou envie um PR com melhorias de UI/UX, validações, testes ou suporte a novos modelos.
