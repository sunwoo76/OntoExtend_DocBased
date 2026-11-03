# 문서 기반 온톨로지 확장 (doc 파이프라인)

> 상태(2026-11-03): 구현과 오프라인 테스트 완료. **실제 LLM으로는 아직 한 번도 돌려보지 않았다** (작업 환경에 API 키가 없음). 프롬프트 품질은 실제 실행 후 조정이 필요하다.

## 1. 무엇을 하나

파싱된 문서를 받아 다음 순서로 온톨로지를 키운다.

```
파싱된 문서 (.md/.txt/.json/.jsonl)
  → 청크 분할 (헤딩 경계 유지, 토큰 상한)
  → 청크마다 CQ 생성 (LLM, 근거 인용 포함)
  → 근거 검증 + 중복 CQ 제거
  → CQ마다 기존 OntoExtend 확장 (검색 → 조각 생성 → 검증/재시도)
  → 작업 온톨로지에 병합 + 새/변경 요소 재색인
  → 다음 CQ / 다음 청크
```

청크는 문서 순서대로 처리하고, 각 청크의 CQ로 바로 온톨로지를 확장한다. 그래서 뒤 청크는 앞 청크가 만든 클래스와 속성을 검색해서 재사용할 수 있다. 시작 온톨로지는 있어도 되고(`--onto`/`--onto-dir`) 없어도 된다.

## 2. 실행

```bash
cd "OntoExtend Script"

# 빈 온톨로지에서 시작
python rag-ontoextend.py doc --file ../Dataset/sample_docs/textile_circularity_sample.md \
  --base-iri http://example.org/doc-extension# --max-retries 2 --run-dir ../runs/doc_empty

# base 온톨로지에서 시작 (정답 온톨로지가 아니라 다른 방법론으로 만든 출발점)
python rag-ontoextend.py doc --file parsed.json --onto-dir ../Dataset/OntoDESIDECoreOntology \
  --base-iri http://example.org/doc-extension# --max-retries 2 --run-dir ../runs/doc_base

# CQ만 뽑아서 먼저 눈으로 확인 (확장 안 함). 결과 CSV는 csv 명령에 그대로 넣을 수 있다.
python rag-ontoextend.py doc --file parsed.md --cq-only --run-dir ../runs/cq_check
python rag-ontoextend.py csv --file ../runs/cq_check/generated_cqs.csv --run-dir ../runs/from_csv
```

`scripts/run_doc_sample.sh empty|base`가 위 첫 두 명령을 감싼다.

주요 옵션:

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--chunk-tokens` | 400 | 청크당 최대 토큰 |
| `--chunk-overlap` | 0 | 다음 청크 앞에 반복할 이전 문단 토큰 수 |
| `--rechunk` | 끔 | JSON 입력의 청크 경계를 무시하고 다시 나눔 |
| `--max-cqs` | 5 | 청크당 최대 CQ 수 |
| `--dedup-threshold` | 0.92 | 이 코사인 유사도 이상이면 앞선 CQ의 중복으로 보고 버림 |
| `--keep-ungrounded` | 끔 | 근거 인용이 청크에 없는 CQ도 유지 |
| `--no-story` | 끔 | 청크 본문을 확장 프롬프트의 ONTOLOGY STORY로 넘기지 않음 |
| `--base-iri` | 참조 온톨로지의 기본 네임스페이스, 없으면 `http://example.org/ontoextend#` | 새 요소를 넣을 `:` 네임스페이스 |
| `--max-retries` | 0 | 검증 실패 시 오류를 LLM에 돌려주고 다시 생성하는 횟수 |

`cq`, `csv` 명령에도 `--base-iri`, `--max-retries`가 붙었고, `--onto`를 빼면 빈 온톨로지에서 시작한다.

### 입력 형식

파서 결과 형식을 아직 모르므로 흔한 형태를 모두 받는다.

- `.md` / `.txt`: Markdown 헤딩(`#`)을 섹션 경계로 쓰고 청크에 헤딩 경로를 남긴다. 코드 블록 안의 `#`, HTML 주석은 무시한다.
- `.json`: 청크 리스트, 또는 `chunks`/`sections`/`pages`/`elements`/`blocks`/`items` 리스트를 가진 객체. 항목은 문자열이거나 `text`/`content`/`markdown`/`md`/`body` 필드를 가진 객체이고, `id`, `heading`(또는 `title`/`section`/`heading_path`), `page`가 있으면 쓴다.
- `.jsonl`: 줄마다 위 객체 하나.

JSON의 청크 경계는 그대로 쓰고, 상한의 1.5배를 넘는 항목만 쪼갠다.

## 3. 출력 (`--run-dir` 안)

| 파일 | 내용 |
|---|---|
| `chunks.jsonl` | 청크 목록 (id, 순서, 헤딩 경로, 본문) |
| `generated_cqs.csv` | 모든 CQ 후보와 상태(`kept`/`duplicate`/`ungrounded`), 근거 인용, 확장 성공 여부, 새/변경 요소 수. `CQ`, `CQID`, `Story`, `chunk_id` 열이 있어 `csv` 명령 입력으로 재사용 가능 |
| `<문서>_<태그>_updated.ttl` | **갱신된 온톨로지** (시작 온톨로지 + 모든 확장) |
| `<문서>_<태그>_combined.ttl` | 확장분만 (시작 온톨로지에 없던 트리플) |
| `working_ontology.ttl` | CQ 하나 처리할 때마다 갱신되는 현재 상태 |
| `element_provenance.csv` | 요소별로 어느 청크·CQ에서 `new`/`changed` 되었는지 |
| `duplicate_candidates.csv` | 새 요소가 기존 요소와 라벨이 같을 때 (자동 병합은 하지 않음) |
| `retrieved_elements/*.csv` | CQ별 검색 결과, 이제 `rank`, `score` 포함 |
| `cq_processing_log.csv` | 기존 열 + `attempts`, `new_elements`, `changed_elements`, `chunk_id` |
| `document_run_summary.json` | 청크 수, 후보/유지/중복/근거없음 CQ 수, 확장 성공/실패 수 |

## 4. 핵심 변경 (공통 기반, PLAN 2단계 핵심)

1. **작업 그래프 병합.** 조각을 텍스트로 이어 붙이던 방식 대신 `WorkingOntology`가 rdflib 그래프 하나를 유지한다. 트리플 단위로 병합하므로 기존 클래스를 다시 선언한 조각은 실제로 새 공리만 보탠다. 빈 노드 구조(제약 등)는 구조 서명으로 비교해서, 같은 제약을 반복해도 중복으로 쌓이지 않는다.
2. **새 요소 / 변경 요소 구분.** 병합 전에 주어로 존재하던 IRI에 트리플이 붙으면 `changed`, 처음 나온 IRI면 `new`.
3. **재색인.** `FaissVectorStore`를 `IndexIDMap2` 기반으로 바꿔 URI의 벡터를 교체할 수 있게 했다. 새 요소와 변경 요소는 병합된 그래프에서 다시 추출·임베딩되므로, 다음 CQ 검색 때 갱신된 스니펫이 보인다. (이전에는 이미 본 URI는 건너뛰었다.)
4. **빈 상태 시작.** `--onto` 없이 실행 가능. `:` 접두어가 항상 선언되고, 검색 결과가 없으면 "아직 요소가 없으니 `:`에 새로 만들라"는 문구가 들어간다. `--base-iri`를 주면 프롬프트가 새 요소를 `:`에 두도록 지시하고, 참조 온톨로지의 기본 네임스페이스는 `core:`로 남는다.
5. **자기 교정.** `--max-retries N`이면 Turtle 파싱 실패, domain/range 누락, 빈 조각일 때 오류 메시지와 이전 답을 붙여 다시 요청한다. 접두어 선언을 빠뜨린 조각은 프롬프트의 접두어 블록을 붙여 한 번 더 파싱한다.
6. **검색 점수 로깅.** 검색 결과 CSV에 순위와 점수를 남긴다 (PLAN의 τ 실험 준비).

### 원래 동작과 달라진 점

- `--max-retries 0`, `--base-iri` 없음이면 프롬프트와 검색은 원래와 같다. 단, 검색 결과가 0개일 때와 이야기(Story)가 있을 때만 프롬프트에 문구가 추가된다.
- `*_combined.ttl`은 이제 조각 텍스트를 이어 붙인 것이 아니라 **시작 온톨로지에 없던 트리플만** 담은 그래프 직렬화다. 시작 온톨로지까지 포함한 결과는 `*_updated.ttl`.
- 성공한 조각의 요소는 조각 파일이 아니라 작업 그래프(`working_ontology`)를 출처로 색인된다. 기존 요소가 바뀐 경우 원래 출처 이름은 유지한다.

## 5. 설계 판단과 근거

- **CQ 근거 인용 필수.** 청크에 없는 내용을 지어낸 CQ를 막는 가장 싼 방법이다. 인용이 청크에 없으면 버린다(공백·구두점 차이와 작은 수정은 허용).
- **CQ는 영어로.** 기존 프롬프트, 참조 온톨로지 라벨, 임베딩 검색이 모두 영어 기준이다. 근거 인용은 원문 언어 그대로 둔다. 한국어 문서도 이 방식으로 처리할 수 있지만, 라벨 언어 정책은 결정이 필요하다 (아래 열린 질문).
- **청크 본문을 ONTOLOGY STORY로 전달.** 원래 템플릿이 OS를 받도록 되어 있었지만 채워지지 않았다. CQ만으로는 모호한 용어("trained person")를 청크 문맥이 정해 준다. 비교 실험용으로 `--no-story`가 있다.
- **중복 후보는 기록만.** 자동 병합은 평가 지표를 왜곡할 수 있어 PLAN대로 로그만 남긴다.
- **base 온톨로지는 정답이 아님.** 그래서 평가에서 "base와의 정렬"을 품질 상한으로 쓰지 않고, CQ 답변 가능성(SPARQL 통과율)과 구조 지표를 주 지표로 둔다 (PLAN 1단계를 이 기준으로 진행).

## 6. 테스트

```bash
pip install pytest
python -m pytest tests -q          # 15개, API 없이 실행
python tests/smoke_offline.py --limit 13   # 원래 csv 파이프라인 회귀 확인
```

테스트가 확인하는 것: 벡터 교체, 빈 시작과 접두어, 새/변경 요소 구분과 재색인, 같은 제약의 중복 방지, 확장분만 출력, 중복 라벨 기록, 재시도 루프, 잘못된 Turtle 처리, 청크 분할(헤딩·크기·겹침·JSON/JSONL), 근거 검증, 문서 파이프라인 전체(중복·근거없음 CQ 제거, Story 전달, 앞 청크 요소 재사용, 출처 기록), `--cq-only`.

`Dataset/sample_docs/textile_circularity_sample.md`는 테스트용으로 직접 쓴 **합성 문서**다. 실험에는 실제 파싱 결과를 써야 한다.

## 7. 다음 단계

1. 실제 API 키로 샘플 문서를 `empty`/`base` 두 모드로 돌려 CQ 품질과 확장 결과를 확인하고 프롬프트를 조정.
2. 실제 파서 출력 형식이 정해지면 로더를 맞춤 (지금은 범용 로더).
3. PLAN 1단계 평가 도구: CQ 계약 → ABox + SPARQL 통과율, 구조 지표 (HermiT), 재사용 지표. 문서 모드에서는 CQ 자체의 품질(근거 비율, 중복률, 청크 커버리지)도 지표에 포함.
4. 순서 민감도: 청크 순서를 섞어 반복 실행.

### 열린 질문

1. 문서가 한국어일 때 새 요소의 `rdfs:label` 언어 (영어만 / 원문 언어 병기 `@ko`).
2. CQ를 청크 단위로 바로 확장(현재)할지, 문서 전체 CQ를 먼저 모아 정렬·병합한 뒤 확장할지. 후자는 순서 민감도가 줄 수 있어 비교 조건 후보.
3. 청크 크기 기본값(400 토큰)과 청크당 CQ 수(5)는 임의의 출발점이다.
