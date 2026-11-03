# 빈 온톨로지에서 시작하는 OntoExtend 확장 계획 (M1 vs M2)

> 기준 코드: `OntoExtend Script/rag-ontoextend.py` (업스트림 `dersuchendee/OntoExtend` `71b593d`와 동일).
> 이 문서는 README 기반으로 세운 초기 계획을 **실제 코드를 확인한 뒤** 보정한 버전이다.

## 1. 목표와 연구 질문

참조 온톨로지 없이 CQ만으로 온톨로지를 시작해 OntoExtend 방식으로 점진적으로 키우는 모드(M2)를 추가하고, 참조 온톨로지 기반 원래 방식(M1)과 비교한다.

- **RQ1.** CQ만으로 점진 생성한 온톨로지(M2)는 참조 온톨로지 기반(M1)에 비해 얼마나 근접한 품질을 내는가?
- **RQ2.** M2의 결과는 CQ 처리 순서에 얼마나 민감한가?
- **RQ3 (선택).** 유사도 임계값 τ를 추가하면 M2의 부적절한 재사용과 중복이 줄어드는가?

## 2. 코드 확인 결과 (README 가정과의 차이)

| # | README/초기 계획의 가정 | 실제 코드 | 계획에 미치는 영향 |
|---|---|---|---|
| F1 | 저장소가 업스트림과 다를 수 있음 | 파일 단위로 업스트림과 **완전히 동일**. "DocBased" 관련 변경은 아직 없음 | 모든 확장을 이 저장소에서 새로 시작 |
| F2 | 임계값이 숨어 있을 수 있음 | **없음.** `FaissVectorStore.search()`가 점수를 돌려주지만 `_retrieve_relevant()`에서 `for el, _ in results`로 버림 (L929-938). 출처 필터만 있음 | τ는 순수 추가 요소. 점수 로깅도 새로 넣어야 함 |
| F3 | `--include-cache`를 켜야 CQ n의 조각이 CQ n+1에서 검색됨 | **항상 검색됨.** `run_csv()`는 성공한 조각을 `origin="session"`으로 색인하고, `_allowed_sources()`는 `session_sources`를 무조건 포함 (L821-826, L1048). `--include-cache`는 **이전 실행들**이 `~/.core_ontology_rag_cache/individual_ontologies/`에 남긴 조각까지 불러오는 옵션 | M2에서 `--include-cache` "강제"는 불필요. 오히려 켜면 다른 실험의 조각이 섞여 들어오므로 **끄고, 실행마다 캐시 디렉터리를 분리**해야 함. M1도 이미 자기 조각을 재사용하는 구조임에 유의 |
| F4 | 참조 온톨로지만 빼면 돌아감 | `expand_onto_args()`가 파일이 없으면 `sys.exit(1)`. `PrefixManager`가 기본 `:` 네임스페이스를 못 찾고, 프롬프트는 "없으면 `:`를 쓰라"고 하지만 prefix 블록에 `:` 선언이 없어 파싱 실패 가능. 결합본 헤더는 `<http://www.example.org/ontology>` 하드코딩 | `--base-iri`로 `@prefix :` 선언을 주입해야 M2가 동작함 |
| F5 | 검증 실패는 실패로 기록 | domain/range 누락 시 `ValueError` → 조각 **전체 폐기, 색인도 안 됨** | M2에서는 초반 CQ 하나의 실패가 이후 전체에 구멍을 냄. 재시도(자기 교정) 루프의 가치가 M1보다 큼 |
| F6 | 새 요소만 재색인 | `_index_elements()`가 이미 본 URI는 건너뜀 (`is_new`). 기존 클래스에 공리를 덧붙인 조각이 와도 **스니펫이 갱신되지 않음** | 성장하는 그래프 구조에서는 "변경된 요소 재색인(벡터 교체)"이 필요. FAISS `IndexFlatIP`는 삭제가 불편하므로 `IndexIDMap` 사용 또는 CQ마다 재구축 검토 |
| F7 | 조각 병합은 텍스트 연결 | 맞음 (`OntologyCombiner.combine`). 각 조각의 `@prefix` 줄만 제거하고 이어 붙임 | 2단계에서 rdflib Graph 병합으로 교체 |
| F8 | temperature=0 고정 | 기본값 0.7. env에서 읽으면 **문자열**로 넘어감 (`LLM_TEMPERATURE = os.getenv(..., 0.7)`) | `float()` 변환 + 실험용 0 고정. 추론형 모델(o1/o4 계열)은 temperature 미지원일 수 있음 |
| F9 | 로그 CSV 재사용 | `generation_runs.csv`, `ontology_elements.csv`는 매 실행 **덮어쓰기**, 나머지는 **append**. 모두 전역 캐시 디렉터리 하나에 쌓임 | `--run-dir` 같은 실행별 출력 경로가 필요 (M2의 순서별 반복 실행 때문에 필수) |
| F10 | EU-project / Industry CQ 세트로 실험 | 저장소에는 섬유 도메인 CQ 13개(`Dataset/pre expriment data/RAG/testing-data/cqs.csv`)만 있음. EU/Industry 세트는 README 예시 5개씩뿐 | **CQ 세트 확보가 선행 조건** |
| F11 | 의존성 | `requirements.txt`는 `openai==0.27.8`, `pandas==1.5.3`인데 코드는 `openai.AsyncOpenAI`(v1 API) 사용. `pyproject.toml`은 `openai>=1.108`, `pandas>=2.3.2` | 0단계에서 `pyproject.toml` 기준으로 환경 구성 |
| F12 | OWL vs SHACL | 프롬프트 지시 5번이 "참조 온톨로지가 OWL 제약을 쓰면 OWL, SHACL만 쓰면 SHACL"로 되어 있어 참조 온톨로지에 의존 | M2에는 근거가 없음 → 두 모드 모두 스타일을 명시적으로 고정해야 비교가 공정 |

## 3. 실험 조건

| 조건 | 초기 온톨로지 | 검색 방식 | 반복 |
|---|---|---|---|
| M1 | OntoDESIDE (`Dataset/OntoDESIDECoreOntology`) | top-K (원래 방식), 세션 조각 재사용 포함 | 1회 |
| M2 | 없음 (`--base-iri`만) | top-K, 세션 조각 재사용 (원래와 동일) | CQ 순서 3~5가지 |
| M2-τ (선택) | 없음 | top-K + τ 필터 | M2와 같은 순서들 |

공통 고정값: LLM, 임베더(`text-embedding-3-small`), temperature=0, top-K=20, 프롬프트(2단계 버전), 제약 스타일(OWL/SHACL). 실행마다 별도 캐시 디렉터리, `--include-cache` 끔. M1과 M2의 차이는 초기 온톨로지 유무 하나뿐이다.

## 4. 평가 지표 (전부 자동)

평가 LLM은 생성 LLM과 다른 계열로 고정한다.

1. **CQ 계약 통과율.** CQ마다 조건·기대 답·예제 사실·반례를 담은 계약을 온톨로지를 보지 않은 상태에서 생성하고 내부 일관성 검사 후 실험 전에 고정한다. 평가 시 결과 온톨로지의 어휘로 ABox와 SPARQL을 만들어 실행하고, 기대 답이 나오고 반례가 배제되면 통과. 없는 IRI 사용은 실패. pass@3. 실패는 "개념 부재" / "번역 실패"로 구분.
2. **참조 온톨로지 정렬.** 결과의 클래스·속성을 OntoDESIDE와 라벨·임베딩으로 매칭한 대응 비율. M1은 상한선으로 해석.
3. **구조 지표.** HermiT(owlready2) 일관성(unsatisfiable 클래스 수), 중복 개념 수, 고립 클래스 수, domain/range 충돌 수.
4. **재사용 지표.** 재사용률 + 평가 LLM이 판정한 부적절 재사용률 (RQ3 핵심).
5. **순서 민감도 (M2).** 순서별 지표 분산, 결과 온톨로지 간 쌍별 유사도.
6. **비용·시간.** 기존 `token_cost_log.csv`, `generation_runs.csv`.

## 5. 구현 단계 (코드 기준으로 보정)

### 0단계: 환경 및 베이스라인 재현
- `pyproject.toml` 기준 환경 구성 (F11). `LLM_TEMPERATURE` float 변환 (F8).
- 실행별 출력 경로 옵션 추가(`--run-dir`, 기본값은 기존 경로 유지) — 로직 변경 없이 로그 분리만 (F9).
- 섬유 CQ 13개 + OntoDESIDE로 원래 파이프라인 실행. 결과를 회귀 기준으로 보관.
- EU-project / Industry CQ 세트 확보 (F10).

**진행 상황 (2026-11-03)**
- 완료: `LLM_TEMPERATURE`/`VECTOR_DIM` 형 변환, `--run-dir` 옵션(`cq`, `csv` 둘 다), `requirements.txt`의 openai·pandas 버전 수정, `scripts/run_baseline.sh`, 오프라인 스모크 테스트 `tests/smoke_offline.py`.
- 스모크 테스트는 해시 임베더와 고정 응답 LLM으로 실제 `run_csv()`를 끝까지 돌린다. 섬유 CQ 13개 전부 통과, 출력이 `--run-dir`에만 쓰이는 것, `--include-cache` 없이도 앞 CQ 조각이 검색되는 것(F3)을 확인했다. 실행: `python tests/smoke_offline.py --limit 13`.
- 남음: 실제 API 키로 `scripts/run_baseline.sh` 실행 (`.env`에 `OPENAI_API_KEY`, `LLM_TEMPERATURE=0`), EU-project/Industry CQ 세트 확보.

### 1단계: 평가 도구 (`eval/` 독립 스크립트)
- `contracts.py`: CQ 계약 생성·검증·고정 (JSON).
- `cq_eval.py`: 계약 → ABox + SPARQL 생성·실행, pass@k, 실패 유형 분류.
- `alignment.py`, `structure.py` (HermiT 포함), `reuse.py`.
- 0단계 결과에 먼저 적용해 도구 자체를 검증.

### 2단계: 공통 구조 리팩터링
- `OntologyCombiner` 텍스트 연결 → **하나의 rdflib Graph**를 유지하며 매 CQ 결과를 병합 (F7). 최종 출력은 그래프 직렬화.
- `--base-iri`(+ 접두어 이름)로 `@prefix :` 선언을 프롬프트 prefix 블록과 결합본 헤더에 주입 (F4). 명명 규칙(CamelCase 클래스, lowerCamel 속성) 명시.
- `--init existing|incremental` 플래그. `incremental`이면 `--onto` 없이 실행 허용.
- 새 요소 추적: 병합 전후 그래프의 주어 IRI 차집합. 기존 IRI에 공리가 붙은 경우는 "변경 요소"로 분리.
- 변경 요소 재색인 (F6): `FaissVectorStore`를 `IndexIDMap` 기반으로 바꾸거나 URI→행 매핑으로 벡터 교체.
- 검색 점수 로깅 (F2): `_retrieve_relevant()`가 `(element, score)`를 유지하고 `retrieved_elements/*.csv`에 `score`, `rank` 열 추가.
- 프롬프트: "관계에 속성이 붙으면 중간 개체(reification)로 모델링" 지시, 새 요소별 생성 근거 출력, OWL/SHACL 스타일 고정 (F12).
- 자기 교정 루프(선택, 두 모드 공통): 파싱 오류·domain/range 누락 메시지를 되돌려 최대 N회 재생성 (F5). 넣는다면 M1·M2 모두에 동일 적용.
- 프롬프트가 바뀌므로 M1을 이 버전으로 재실행. 0단계 결과는 회귀 검사용으로만 사용.

**진행 상황 (2026-11-03, 핵심만 먼저)**
- 순서 변경: 문서 기반 확장이 최종 목표이므로 2단계 핵심 → 문서 청크→CQ 파이프라인 → 1단계 평가 도구 순으로 진행한다. 자세한 내용은 `docs/DOC_PIPELINE.md`.
- 완료: rdflib 그래프 병합(`WorkingOntology`, 빈 노드 구조 비교로 중복 제약 방지) (F7), `--base-iri` 및 참조 온톨로지 없이 실행 (F4), 새/변경 요소 구분과 `IndexIDMap2` 기반 재색인 (F6), 검색 점수·순위 로깅 (F2), `--max-retries` 자기 교정 루프(기본 0) (F5), 요소 출처 로그, 중복 라벨 후보 로그, 빈 검색 결과용 bootstrap 문구.
- 미완: 새 요소별 생성 근거 출력, reification 지시, OWL/SHACL 스타일 고정 (F12) — 실제 LLM 출력을 본 뒤 프롬프트와 함께 조정.
- 테스트: `python -m pytest tests -q` (15개, 오프라인).

### 문서 기반 확장 (최종 목표, 추가)
- 파싱된 문서 → 청크 → 청크별 CQ(근거 인용 필수, 중복 제거) → CQ별 확장. `rag-ontoextend.py doc` 명령.
- 시작 온톨로지는 있을 수도 없을 수도 있음. 있을 때도 사람이 만든 정답이 아니라 다른 방법론으로 만든 base 온톨로지이므로, 평가에서 base와의 정렬은 상한이 아니라 참고 지표로만 쓴다.
- M1/M2 구분이 그대로 적용됨: base 있음 = M1형, 없음 = M2형. CQ 순서 민감도(RQ2)는 청크 순서 민감도가 됨.

### 3단계: M2 구현 및 실행
- 첫 CQ(또는 검색 결과가 비어 있을 때)용 **bootstrap 프롬프트**: base IRI, 명명 규칙, 최상위 클래스 구조를 세우라는 지시. 인덱스가 비면 `store.search()`가 `[]`를 돌려주므로 분기 조건은 "검색 결과 0개"로 충분.
- 병합 시 중복 검사: 라벨 정규화 일치 + 임베딩 유사도로 중복 후보 탐지, 로그로 남김(자동 병합은 하지 않음 — 지표 왜곡 방지).
- CQ 순서 3~5가지(원래 순서 + 시드 고정 셔플)로 실행 → RQ1, RQ2.
- (변형, 선택) 전체 CQ를 먼저 보여주고 골격 온톨로지를 만든 뒤 진행하는 하이브리드. 순서 민감도가 크게 나올 때 비교 조건으로 추가.

### 4단계 (선택): M2-τ
- 3단계에서 부적절 재사용·중복이 눈에 띌 때만 진행.
- 2단계에서 로깅한 점수 분포로 τ 후보 1~3개 선정, `_retrieve_relevant()`에 `score >= τ` 필터 추가 (`--tau`, 기본 없음).
- 필터 후 결과가 비면 bootstrap 프롬프트로 분기. M2와 같은 순서로 실행 → RQ3.

## 6. 체크포인트

- 1단계 완료: M1 기준 수치 확보 → 평가 도구 동작 판단.
- 3단계 완료: RQ1·RQ2 결과 → 4단계 진행 여부 결정.

## 7. 열린 질문

1. EU-project / Industry CQ 세트(와 각 세트의 참조 온톨로지)의 위치. 저장소에는 섬유 CQ 13개만 있음.
2. 생성 LLM과 평가 LLM 선택 (생성: o1-preview / GPT-5 / 기타, 평가: 다른 계열).
3. 제약 스타일을 OWL 제약으로 고정할지 SHACL로 고정할지.
4. 자기 교정 루프를 기본 조건에 넣을지, 별도 변형으로 둘지.
