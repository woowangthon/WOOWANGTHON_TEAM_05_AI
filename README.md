# 온보딩 — Gemma 추론 게이트웨이

2026 AI 우왕톤 05팀 **우짤래**. Google AI Studio의 **Gemma 4 무료 API** 또는 로컬 Ollama를 사용하는 FastAPI 서비스입니다. 기존 `/interpret`, `/explain` API로 백엔드의 조건 해석과 설명 생성을 지원합니다.

## Google AI Studio 키로 실행

Python 3.12 이상과 [uv](https://docs.astral.sh/uv/)가 필요합니다. Ollama나 로컬 GPU는 필요하지 않습니다.

이 디렉터리에서 `.env.example`을 `.env`로 복사하고 발급받은 키를 `GEMINI_API_KEY`에 입력합니다. PowerShell에서는 `Copy-Item .env.example .env`를 사용하세요. 이미 `.env`가 있으면 복사하지 말고 기존 파일을 수정합니다.

```dotenv
AI_PROVIDER=google
GEMINI_API_KEY=여기에_AI_Studio_키_입력
GEMMA_MODEL=gemma-4-26b-a4b-it
AI_API_KEY=
```

`GEMINI_API_KEY`는 Google 호출용 비밀키이고, `AI_API_KEY`는 BE → AI 서버 사이의 인증용으로 별개입니다. Google 키는 AI 서버의 `.env`에만 보관합니다. `.env`는 Git과 Docker 빌드에서 제외되며, `.env.example`과 프론트엔드에는 실제 키를 넣지 마세요.

```sh
uv sync --locked --group dev
uv run python -m uvicorn app.main:app --host 127.0.0.1 --port 8001
```

AI 저장소의 `.env`를 로드하며 이미 설정된 프로세스 환경변수가 우선합니다. 변경 후 서버를 재시작하세요.

- API 문서: http://127.0.0.1:8001/docs
- 준비 확인: http://127.0.0.1:8001/health
- 정상 결과: `{"status":"ready","provider":"google","model":"gemma-4-26b-a4b-it"}`

`/health`는 모델 메타데이터만 조회하며 생성 토큰을 사용하지 않습니다. 실제 추론 성공이나 남은 생성 한도를 보장하지 않으므로 `/docs`의 `/interpret`에서 다음 요청으로 확인하세요.

```json
{"message":"더 저렴하게, 통근 60분","profile":{"max_commute":45}}
```

`AI_API_KEY`를 설정했다면 요청 헤더에 같은 `X-API-Key`가 필요합니다.

## 백엔드 연결

형제 폴더 `WOOWANGTHON_TEAM_05_BE`의 `.env`를 설정하고 별도 터미널에서 실행합니다.

```dotenv
AI_SERVICE_URL=http://127.0.0.1:8001
AI_API_KEY=
BACKEND_API_KEY=
CORS_ORIGINS=http://localhost:3000
```

```sh
uv sync --locked --group dev
uv run python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

인증을 사용하면 BE와 AI의 `AI_API_KEY` 값을 동일하게 설정합니다. 프론트엔드 실행은 [FE README](../WOOWANGTHON_TEAM_05_FE/README.md)를 따릅니다. BE의 `AI_SERVICE_URL`이 비어 있으면 규칙 기반 데모가 실행됩니다.

Docker는 BE 폴더에서 `docker compose up --build -d`로 실행합니다. AI 폴더의 `.env`를 컨테이너에 전달하며 Ollama는 기본 실행에서 제외됩니다. `env_file.required`를 지원하는 Docker Compose 2.24.0 이상이 필요합니다. Docker의 서버 간 `AI_API_KEY`는 BE 폴더의 `.env` 값으로 통일됩니다.

## 무료 범위와 오류 처리

[Google 공식 요금표](https://ai.google.dev/gemini-api/docs/pricing#gemma-4)에서 Gemma 4 입력·출력은 무료로 제공됩니다. 프로젝트별 요청 한도와 모델 접근 가능 여부는 [AI Studio](https://aistudio.google.com/)에서 확인하세요. 무료 서비스의 입력·출력은 Google 제품 개선에 사용될 수 있습니다. 요금과 한도는 변경될 수 있습니다.

- Google 경로는 `gemma-4-26b-a4b-it`, `gemma-4-31b-it`만 허용합니다.
- 한도 초과(429)는 안내와 함께 503을 반환합니다. 자동 재시도, 다른 모델 전환, 유료 도구 호출은 없습니다.
- 키 누락, 권한 오류, 사용할 수 없는 모델은 503으로 안내하며 키나 원본 API 오류 본문은 반환하지 않습니다.
- BE의 조건 변경 실패는 기존 오류 처리로 전달되고, 설명 실패는 데모 결과와 경고로 표시됩니다.

## 로컬 Ollama 사용

API 키 없이 로컬 추론하려면 [Ollama](https://ollama.com/)를 설치하고 `.env`를 변경합니다. 기존 Ollama 설정을 사용하는 경우에도 `AI_PROVIDER=ollama`를 명시하세요.

```dotenv
AI_PROVIDER=ollama
OLLAMA_BASE_URL=http://127.0.0.1:11434
GEMMA_MODEL=gemma3:4b
AI_API_KEY=
```

```sh
ollama pull gemma3:4b
ollama serve
```

이미 Ollama가 실행 중이면 `serve`는 다시 실행하지 않습니다. FastAPI 실행 명령은 위와 같습니다. Docker에서는 AI `.env`를 위처럼 변경한 뒤 BE 폴더에서:

```sh
docker compose --profile local up --build -d
docker compose exec ollama ollama pull gemma3:4b
```

## 역할과 검증

| API | 역할 | 검증 |
| --- | --- | --- |
| GET `/health` | 공급자 연결과 지정 모델 확인 | 준비되지 않으면 503 |
| POST `/interpret` | 변경 요청에서 명시한 조건 추출 | 허용 필드·범위·자료형 검증 |
| POST `/explain` | 설명 초점과 근거 ID 선택 | budget/commute/checklist와 등록된 ID만 허용 |

JSON Schema를 지시문에 전달하고 Pydantic으로 응답을 검증합니다. Ollama에는 구조화 출력 설정도 전달합니다. Google 응답의 추론 부분은 제외하고 완료된 최종 JSON만 사용합니다. 비용·법률 결론·출처 URL은 모델이 생성하지 않으며 서버가 검증된 데이터와 초점을 문장으로 구성합니다.

응답 제한은 35초, 출력 최대 512토큰입니다. `/interpret`는 수정 가능한 현재 조건만 전송합니다. 정착 계획의 비용·주거·통근 데이터는 시연용입니다.

```sh
uv run python -m pytest -q
```

테스트는 Google과 Ollama를 모킹하여 요청 계약, JSON·출처 검증, 인증, 한도·설정 오류를 확인합니다. 실제 API 추론 품질·지연과 Docker 실행은 별도로 확인해야 합니다.

- [Google Gemma API 문서](https://ai.google.dev/gemma/docs/core/gemma_on_gemini_api)
- [Ollama 구조화 출력](https://docs.ollama.com/capabilities/structured-outputs)
- [Backend](https://github.com/woowangthon/WOOWANGTHON_TEAM_05_BE)
- [Frontend](https://github.com/woowangthon/WOOWANGTHON_TEAM_05_FE)
