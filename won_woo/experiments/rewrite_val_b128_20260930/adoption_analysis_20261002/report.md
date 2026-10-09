# Val rewrite adoption and semantic review (2026-10-02)

Read all 10,895 records. Accepted 1,290 (11.84%). Rechecked all accepted queries with the actual local RoBERTa tokenizer and stored verifier verdicts. All passed the recorded adoption guards.

## Historical rule compatibility

Run source SHA: 8d4da0dc57f17618206d9be96b3e592cf9c83bbbc08b3b7e1dca309f66e787b0
Current source SHA: bf617d935402746c8173977f63c28f52e01ef031aad87ea5362950f8ba0e4a50
Exact saved original source recovered: True. candidate_check/support_check/verdict_check/normalized/summary_check ASTs unchanged: {'candidate_check': True, 'support_check': True, 'verdict_check': True, 'normalized': True, 'summary_check': True}. Prompt SHA matches: True.

This is a replay of stored LLM verdicts, not a new LLM judgement or an independent semantic certification.

## Train / val comparison

| Item | Train | Val |
|---|---:|---:|
| Total | 87,174 | 10,895 |
| Accepted | 11,064 | 1,290 |
| Acceptance rate | 12.69% | 11.84% |
| Accepted original has none of 3 lexical flags | 4,606 (41.63%) | 588 (45.58%) |
| Shorter rewrites | 7,482 | 858 |
| Grounding includes sibling | 373 | 37 |
| Empty grounding | 12 | 0 |

## Actual adoption requirements

- Changed nonempty single-line query, maximum 22 words and 29 RoBERTa tokens including special tokens.
- Minimum 16 words is soft; shorter faithful candidates are allowed.
- Temporal keyword sequences equal; supplied grounding spans literally match source captions.
- Event/roles/negation/time verdicts all true, no imported sibling event, no unsupported details, valid schema.
- No duplicate among accepted rewrites in the same video.
- Neither applying one of the four content transformations nor retrieval-score gain is required.

## Accepted properties

Words 4–22, mean 10.67; 1,203 below soft minimum 16.
Tokens 7–29, mean 15.40.
Original accepted word mean 12.18; shorter/same/longer=858/288/144.
Target-only grounding queries 1,253; sibling citation present 37; empty 0.
Grounding.text absent from candidate in at least one item: 1,191 queries. This is allowed for paraphrase; not alone evidence of an error.

## Confirmed changes relative to source captions

Purposeful inspection confirmed 8 caption-meaning changes below. These examples are not an exhaustive semantic audit of all 1,290 accepted pairs; no overall error rate is estimated. Actual video frames were not inspected.

| Audit line | Category | Original | Candidate |
|---:|---|---|---|
| 47 | participant_reference | The neighbor corrects Ross about her mother not being gone. | The neighbor tells Ross that his mother is still here. |
| 639 | clothing_attribute | Penny puts both her hands to her chest while listening to the man in a jean shirt. | Penny places her hands on her chest while hearing the man in jeans speak. |
| 1673 | transferred_object | Beckett opens a black folder and hands Jamie Burman a photo. | Beckett opens a black folder and gives it to Jamie Burman with a photo inside. |
| 3439 | head_action | A bald man in a black coat looks down and shakes his head. | A bald man in a black coat lowers his eyes and nods his head in disagreement. |
| 3721 | action_agent | Amy is lying on the couch dressed as a Disney Princess and is trying to get Sheldon to kiss her and he's ignoring her. | Amy lying on the couch as a Disney Princess tries to kiss Sheldon, who ignores her. |
| 6797 | head_action | Penny shakes her head after Raj mentions she's scary. | Penny nods her head in disagreement after Raj comments that she is intimidating. |
| 8160 | action_agent | Amy mentions to Sheldon that he is not alone and he lists off the people that are close to him. | Amy tells Sheldon he is not alone and lists the people around him. |
| 10308 | spatial_relation | Sheldon pushes a green grocery cart behind Penny at the market. | Sheldon pushes a green grocery cart alongside Penny at the market. |

Every case above still has all four preservation verdicts=true, sibling_event_imported=false and unsupported_visual_details=[]. Source citations establish text provenance but do not certify the added/changed candidate meaning.

## Cases needing source/video/event review

- Line 5: Upset changed to looks sad; no explicit expression evidence; breakup causation also altered.
- Line 37: On-face changed to away-face, but a sibling caption supports away-face; target wording ambiguous.
- Line 305: Cracked door changed to broken door; cracked may mean ajar, but source alone is ambiguous.
- Line 453: Chocolate changed to Cinnamon (dog); verifier assumes original typo, requiring source/video review.
- Line 6199: Another caption walking-out event included; matching target event/time not established.

## Supported context rewrite example

Line 397: Raj gives something to Penny while Penny is driving -> Raj hands Penny a cassette tape while she drives. The sibling caption identifies a cassette tape. This is textual grounding for the object; frames/timestamps were not checked.

## Terminal reasons

| Reason | Count |
|---|---:|
| unsupported_summary_span | 3,822 |
| temporal_change | 2,863 |
| negation_equivalent | 1,975 |
| accepted | 1,290 |
| invalid_json | 497 |
| unsupported_visual_details | 141 |
| unchanged | 81 |
| word_budget | 78 |
| temporal_relationships_preserved | 58 |
| unsupported_quote | 38 |
| invalid_verification_schema | 27 |
| token_budget | 15 |
| target_collapse | 5 |
| target_event_preserved | 3 |
| participant_roles_preserved | 2 |

Of 1,975 negation_equivalent rejections, 1,970 originals have no detected negation keyword. This indicates a verifier audit priority, not an automatic count of semantic false rejections.

## Artifacts

- summary.json: full quantitative results and train comparison.
- accepted.csv: all 1,290 accepted original/candidate pairs and verdict explanations.
- semantic_review_cases.json / semantic_review_cases.csv: 8 confirmed caption-meaning changes and 5 cases requiring further review.
- reviewed_error_ids.txt: the 8 confirmed cases only. Excluding these does not certify the remaining dataset.

Same-video sibling context makes this a context-assisted/oracle protocol. No rewrite captions, HDF5 features, checkpoint or original audit file was modified. No new GPU evaluation was run.
