# 온보딩 — Gemma 추론 게이트웨이

2026 AI 우왕톤 05팀 **우짤래**. Ollama의 **Gemma 3**를 호출하는 FastAPI 서비스입니다. 다른 모델로 자동 대체하지 않습니다.

## 로컬 실행

Python 3.12 이상, uv, [Ollama](https://ollama.com/)를 준비합니다. 모델 이용 전 [Gemma 이용 조건](https://ai.google.dev/gemma/terms)을 확인하세요.

```sh
ollama pull gemma3:4b
ollama serve
```

Ollama가 이미 실행 중이면 serve를 다시 실행하지 않습니다. 별도 터미널에서:

```sh
uv sync --locked --group dev
uv run uvicorn app.main:app --host 127.0.0.1 --port 8001
```

기본 모델은 gemma3:4b입니다. 하드웨어에 맞춰 `.env`의 GEMMA_MODEL을 gemma3:1b 등으로 바꿀 수 있습니다. `.env.example`을 복사하고 동일 모델을 Ollama에 받아야 합니다.

```dotenv
OLLAMA_BASE_URL=http://127.0.0.1:11434
GEMMA_MODEL=gemma3:4b
AI_API_KEY=
```

BE에 AI_SERVICE_URL과 동일한 AI_API_KEY를 설정합니다. API 문서는 http://127.0.0.1:8001/docs 에 있습니다. Docker 구성은 BE 저장소의 compose.yaml을 사용합니다.

## 역할과 응답

| API | 역할 | 검증 |
| --- | --- | --- |
| GET `/health` | Ollama 연결과 지정 모델 존재 확인 | 준비가 안 되면 503 |
| POST `/interpret` | 변경 요청과 현재 조건에서 명시된 변경만 추출 | 예산·통근·주거 형태 등 허용 필드, 범위·열거형 검증 |
| POST `/explain` | 도구 결과에서 설명 초점과 관련 출처 ID 선택 | budget/commute/checklist와 등록된 ID만 허용 |

Gemma에 JSON Schema를 전달하고 응답을 Pydantic으로 다시 검증합니다. 비용·법률 결론·출처 URL은 모델이 생성하지 않습니다. BE가 검증한 초점과 서버 소유 데이터를 문장으로 구성합니다. 파인튜닝·자율 도구 선택·벡터 DB는 사용하지 않습니다.

LLM 응답 제한은 35초, 출력 최대 512토큰입니다. 응답 오류·시간 초과·잘못된 JSON은 503을 반환합니다. 이 값은 실제 하드웨어의 지연에 맞춰 조정할 수 있습니다. readiness는 모델 설치 여부이며 실제 추론 품질을 보장하지 않습니다.

## 검증 범위

```sh
uv run pytest -q
```

테스트는 Ollama를 모킹하여 스키마·출처·인증·오류 처리를 검증합니다. **실제 Gemma 추론 품질·지연은 별도 모델 서버에서 검증해야 합니다.** Docker와 GPU 추론은 이 테스트에 포함하지 않습니다.

## 참고

- [Ollama 구조화 출력](https://docs.ollama.com/capabilities/structured-outputs)
- [Ollama Gemma 3 모델](https://ollama.com/library/gemma3)
- [Gemma 이용 조건](https://ai.google.dev/gemma/terms)
- [Backend](https://github.com/woowangthon/WOOWANGTHON_TEAM_05_BE)
- [Frontend](https://github.com/woowangthon/WOOWANGTHON_TEAM_05_FE)
