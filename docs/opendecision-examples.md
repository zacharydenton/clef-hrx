# OpenDecision examples on CLEF

[OpenDecision source](https://github.com/deepanwadhwa/OpenDecision/tree/cbd0b50db42bba6975f342de1789fb3c8f2dc670). 38 requests, 91 questions per model on Radeon 8060S.

## Results

| Model | Precision | Images | GDPR | Insurance: selected evidence | Insurance: full document | Composed decisions |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| CLEF | BF16 | 26/26 | 10/10 | 17/17 | 17/17 | 10/10 |
| Flash | BF16 | 26/26 | 10/10 | 16/17 | 17/17 | 10/10 |
| Omni | INT4 | 26/26 | 10/10 | 17/17 | 17/17 | 10/10 |

OpenDecision's saved gallery matches **24/26** labelled answers. Its published document results are 9/10 GDPR, 17/17 insurance facts and 10/10 composed decisions. These are saved upstream results, not a local rerun of OpenDecision.

Five image questions have no verified label and are unscored. All 17 insurance fact labels are `supports`; that track does not test discrimination between relation classes. The composed rules retain the two upstream deterministic facts (treatment within 24 hours and unknown rental necessity). Subjective scores and unlabelled text examples are shown below.

## Speed

| Model | Overall input tok/s | Text median ms | Image median ms | Image input tok/s | GDPR ms | Full insurance ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CLEF | 467.2 | 370.7 | 1,289.9 | 366.9 | 20,563.0 | 10,571.1 |
| Flash | 962.5 | 118.0 | 648.3 | 535.0 | 6,049.1 | 3,493.4 |
| Omni | 1,204.1 | 135.5 | 812.3 | 551.4 | 4,412.2 | 1,707.8 |

One warmup and one timed repeat per request in a resident process. Timing includes encoding/media preparation, GPU execution and readback; loading is excluded. Every timed output matches its warmup exactly. Runs include intermittent GPU contention. GPU clients and memory telemetry are saved with each run. Single repeats are illustrative latency measurements.

OpenDecision reports median HTTP times from five requests on Apple M3; the gallery uses a small SigLIP2 image model. Its hardware, timing boundary and image preprocessing differ from these native CLEF runs.

## Answers

| Example / question | Expected | CLEF | Flash | Omni |
| --- | --- | --- | --- | --- |
| m1_smoke_test.py:0 / decision | — | isp | isp | isp |
| m1_smoke_test1.py:1 / decision | — | true | true | true |
| m1_smoke_test1.py:2 / decision | — | true | true | true |
| m1_smoke_test1.py:3 / decision | — | 1.986 | 1.979 | 1.982 |
| m1_smoke_test1.py:4 / decision | — | sales | sales | sales |
| sdk_customer / department | — | billing | billing | billing |
| sdk_customer / urgent | — | true | true | true |
| sdk_customer / frustration | — | 0.923 | 1.693 | 1.415 |
| customer_routing / department | billing | billing | billing | billing |
| customer_routing / urgent | true | true | true | true |
| bird / bird_present | true | true | true | true |
| bird / eagle_present | — | true | true | true |
| bird / human_present | false | false | false | false |
| bird / bird_kind | — | hawk | hawk | hawk |
| moon / sky_object | moon | moon | moon | moon |
| moon / plane_present | — | false | false | false |
| moon / tree_present | true | true | true | true |
| cat-patio / cat_present | true | true | true | true |
| cat-patio / cat_count | one | one | one | one |
| cat-patio / cat_napping | false | false | false | false |
| bread / bread_count | one | one | one | one |
| wooded-stream / backyard_flooded | — | false | false | false |
| wooded-stream / water_present | true | true | true | true |
| wooded-stream / water_scene | — | wooded_creek | wooded_creek | wooded_creek |
| patio-animals / animal_count | multiple | multiple | multiple | multiple |
| patio-animals / bear_present | false | false | false | false |
| patio-animals / coyote_present | false | false | false | false |
| patio-animals / animal_present | true | true | true | true |
| patio-animals / animal_type | cat | cat | cat | cat |
| parked-cars / weather | sunny | sunny | sunny | sunny |
| parked-cars / parked_car_count | two | two | two | two |
| cat-window / cat_present | true | true | true | true |
| cat-window / cat_gaze | window | window | window | window |
| bear / animal_present | true | true | true | true |
| bear / animal_type | bear | bear | bear | bear |
| overflowing-bin / bin_present | true | true | true | true |
| overflowing-bin / bin_condition | overflowing | overflowing | overflowing | overflowing |
| overflowing-bin / ground_litter | true | true | true | true |
| damaged-fence / fence_present | true | true | true | true |
| damaged-fence / fence_condition | damaged | damaged | damaged | damaged |
| damaged-fence / fence_material | metal | metal | metal | metal |
| image_api_bear / scene | — | outdoors | outdoors | outdoors |
| image_api_bear / animal | — | true | true | true |
| image_api_bear / brightness | — | 1.444 | 1.282 | 1.166 |
| gdpr / breach_72h | true | true | true | true |
| gdpr / applies_non_eu | true | true | true | true |
| gdpr / dpo_all_orgs | false | false | false | false |
| gdpr / pre_ticked_consent | false | false | false | false |
| gdpr / right_erasure | true | true | true | true |
| gdpr / data_portability | true | true | true | true |
| gdpr / us_federal_law | false | false | false | false |
| gdpr / criminal_penalties | false | false | false | false |
| gdpr / instrument_type | Regulation | Regulation | Regulation | Regulation |
| gdpr / max_fine | TwentyM_or_4pct | TwentyM_or_4pct | TwentyM_or_4pct | TwentyM_or_4pct |
| gdpr / individual_rights | — | 2.899 | 2.873 | 2.891 |
| gdpr / penalty_severity | — | 2.863 | 2.789 | 2.652 |
| gdpr / compliance_burden | — | 2.880 | 2.863 | 2.986 |
| insurance_document / police_rear_impact | supports | supports | supports | supports |
| insurance_document / witness_rear_impact | supports | supports | supports | supports |
| insurance_document / no_acute_fracture | supports | supports | supports | supports |
| insurance_document / no_neurological_deficit | supports | supports | supports | supports |
| insurance_document / discharged_home | supports | supports | supports | supports |
| insurance_document / no_hospitalization | supports | supports | supports | supports |
| insurance_document / prior_neck_stiffness | supports | supports | supports | supports |
| insurance_document / prior_no_imaging | supports | supports | supports | supports |
| insurance_document / prior_no_prescription | supports | supports | supports | supports |
| insurance_document / prior_no_active_treatment | supports | supports | supports | supports |
| insurance_document / employer_two_absences | supports | supports | supports | supports |
| insurance_document / employer_no_wage_calculation | supports | supports | supports | supports |
| insurance_document / pickup_confirmation_outstanding | supports | supports | supports | supports |
| insurance_document / payroll_statement_outstanding | supports | supports | supports | supports |
| insurance_document / emergency_bill_matches_demand | supports | supports | supports | supports |
| insurance_document / therapy_bill_matches_demand | supports | supports | supports | supports |
| insurance_document / claimant_collision_mechanism | supports | supports | supports | supports |

Bold answers differ from the supplied labels. Raw probabilities, timings, composed decisions and all selected-evidence results are in the [JSON report](results/opendecision-20261016.json).

## Reproduction

The adapter reads literal upstream examples without running their model code. Images use the published gallery assets with unchanged question captions. GDPR and insurance receive the complete documents, without retrieval or truncation. Insurance `Relation` is represented as a four-option CLEF `choice`. The generic image API example uses the supplied bear photo. The interactive Doom game has no fixed request corpus and is outside this run.

```sh
git clone https://github.com/deepanwadhwa/OpenDecision.git artifacts/OpenDecision
git -C artifacts/OpenDecision checkout cbd0b50db42bba6975f342de1789fb3c8f2dc670
cargo build --release --locked --features bench-internals
python3 scripts/run_opendecision_examples.py --models clef clef-flash --drop-checkpoint-cache --output tests/results/opendecision-repro
python3 scripts/run_opendecision_examples.py --models clef-omni --precision int4 --drop-checkpoint-cache --output tests/results/opendecision-repro
python3 scripts/report_opendecision_examples.py tests/results/opendecision-repro
```

The runner uses local CLEF and Omni checkpoints at `artifacts/model` and `artifacts/model-omni-tiled`, and the pinned Flash Hugging Face cache. GPU jobs run sequentially under the existing memory guard. The cache-releasing loader uses `CLEF_BENCH_CHECKPOINT_DROP_CACHE=1`; this advises away consumed checkpoint pages without changing weights, precision or kernel selection. Flash used the ordinary build; CLEF and Omni use this loader option. Omni BF16 attempts stopped at the unchanged 1 GiB additional-swap guard before producing predictions; the reported Omni run uses INT4. Raw outputs and telemetry remain in the chosen output directory.
