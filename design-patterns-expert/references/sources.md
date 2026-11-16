# Sources

Every URL the skill cites, grouped by kind, with the files that cite it. "Checked 2026-11-16" means the 2026-11-16 audit fetched the page and machine-matched at least one passage the skill relies on against it (not necessarily every quote from that page) (report: `audits/design-patterns-expert/2026-11-16.md` in the My_Skills repo, local only). "Not checked" means no one has re-read that page against the skill's wording since it was written; treat its quotes as unverified.

Python behaviour claims are checked by execution, not by these pages: `scripts/verify_python_claims.py` runs 20 of them on the current interpreter.

## Python docs and specs (tier 1 for Python behaviour)

| Source | Cited in | Checked |
|---|---|---|
| https://docs.python.org/3/faq/programming.html | python-idioms.md | not checked |
| https://docs.python.org/3/library/abc.html | SKILL.md, python-idioms.md | checked 2026-11-16 |
| https://docs.python.org/3/library/contextlib.html | python-idioms.md | checked 2026-11-16 |
| https://docs.python.org/3/library/dataclasses.html | python-idioms.md | checked 2026-11-16 |
| https://docs.python.org/3/library/functools.html | python-idioms.md | checked 2026-11-16 |
| https://docs.python.org/3/library/typing.html | SKILL.md, more-patterns.md, python-idioms.md | checked 2026-11-16 |
| https://peps.python.org/pep-0544/ | SKILL.md, python-idioms.md | checked 2026-11-16 |
| https://peps.python.org/pep-0562/ | creational.md, python-idioms.md | checked 2026-11-16 |
| https://peps.python.org/pep-0634/ | more-patterns.md | not checked |
| https://peps.python.org/pep-0636/ | more-patterns.md | not checked |
| https://typing.python.org/en/latest/spec/protocol.html | python-idioms.md | checked 2026-11-16 |

## Cloud vendor guidance (tier 1 for that vendor's services; design advice otherwise)

| Source | Cited in | Checked |
|---|---|---|
| https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/ | resilience.md | checked 2026-11-16 |
| https://d1.awsstatic.com/builderslibrary/pdfs/timeouts-retries-and-backoff-with-jitter.pdf | resilience.md | not checked |
| https://docs.aws.amazon.com/prescriptive-guidance/latest/cloud-design-patterns/retry-backoff.html | resilience.md | not checked |
| https://docs.aws.amazon.com/prescriptive-guidance/latest/cloud-design-patterns/transactional-outbox.html | architectural.md | checked 2026-11-16 |
| https://docs.aws.amazon.com/sdk-for-python/v4/guide/config-retries.html | resilience.md | not checked |
| https://docs.stripe.com/api/idempotent_requests | resilience.md | checked 2026-11-16 |
| https://docs.temporal.io/design-patterns/saga-pattern | architectural.md | not checked |
| https://github.com/Netflix/Hystrix/wiki/Operations | resilience.md | checked 2026-11-16 |
| https://learn.microsoft.com/en-gb/azure/architecture/patterns/idempotent-consumer | resilience.md | not checked |
| https://learn.microsoft.com/en-us/azure/architecture/patterns/anti-corruption-layer | structural.md | checked 2026-11-16 |
| https://learn.microsoft.com/en-us/azure/architecture/patterns/bulkhead | resilience.md | checked 2026-11-16 |
| https://learn.microsoft.com/en-us/azure/architecture/patterns/cache-aside | resilience.md | not checked |
| https://learn.microsoft.com/en-us/azure/architecture/patterns/circuit-breaker | resilience.md | not checked |
| https://learn.microsoft.com/en-us/azure/architecture/patterns/rate-limiting-pattern | resilience.md | not checked |
| https://learn.microsoft.com/en-us/azure/architecture/patterns/retry | resilience.md | not checked |
| https://learn.microsoft.com/en-us/azure/architecture/patterns/saga | antipatterns.md, architectural.md | checked 2026-11-16 |
| https://learn.microsoft.com/en-us/azure/architecture/patterns/strangler-fig | architectural.md | checked 2026-11-16 |
| https://learn.microsoft.com/en-us/azure/architecture/patterns/throttling | resilience.md | not checked |

## Standards drafts

| Source | Cited in | Checked |
|---|---|---|
| https://datatracker.ietf.org/doc/html/draft-ietf-httpapi-idempotency-key-header | resilience.md | checked 2026-11-16 |

## Books and engineering writing (signed practitioner sources)

| Source | Cited in | Checked |
|---|---|---|
| https://alistair.cockburn.us/hexagonal-architecture/ | structural.md | checked 2026-11-16 |
| https://blog.ploeh.dk/2011/07/28/CompositionRoot/ | creational.md | checked 2026-11-16 |
| https://blog.ploeh.dk/2015/01/06/composition-root-reuse/ | creational.md | not checked |
| https://lexi-lambda.github.io/blog/2019/11/05/parse-don-t-validate/ | more-patterns.md | not checked |
| https://martinfowler.com/articles/201701-event-driven.html | SKILL.md, architectural.md, behavioral.md | checked 2026-11-16 |
| https://martinfowler.com/articles/feature-toggles.html | resilience.md | checked 2026-11-16 |
| https://martinfowler.com/articles/injection.html | creational.md | not checked |
| https://martinfowler.com/bliki/AnemicDomainModel.html | architectural.md | not checked |
| https://martinfowler.com/bliki/CQRS.html | SKILL.md, architectural.md | checked 2026-11-16 |
| https://martinfowler.com/bliki/StranglerFigApplication.html | architectural.md | not checked |
| https://martinfowler.com/bliki/ValueObject.html | more-patterns.md | not checked |
| https://martinfowler.com/bliki/Yagni.html | SKILL.md | checked 2026-11-16 |
| https://martinfowler.com/eaaDev/EventSourcing.html | SKILL.md, architectural.md | checked 2026-11-16 |
| https://python-patterns.guide/gang-of-four/abstract-factory/ | creational.md | not checked |
| https://python-patterns.guide/gang-of-four/builder/ | creational.md, python-idioms.md | not checked |
| https://python-patterns.guide/gang-of-four/composition-over-inheritance/ | SKILL.md | checked 2026-11-16 |
| https://python-patterns.guide/gang-of-four/decorator-pattern/ | structural.md | not checked |
| https://python-patterns.guide/gang-of-four/factory-method/ | creational.md | not checked |
| https://python-patterns.guide/gang-of-four/flyweight/ | structural.md | not checked |
| https://python-patterns.guide/gang-of-four/iterator/ | more-patterns.md | checked 2026-11-16 |
| https://python-patterns.guide/gang-of-four/singleton/ | creational.md | not checked |
| https://python-patterns.guide/python/module-globals/ | creational.md, python-idioms.md | not checked |
| https://python-patterns.guide/python/prebound-methods/ | python-idioms.md | not checked |
| https://python-patterns.guide/python/sentinel-object/ | more-patterns.md | not checked |
| https://research.google/pubs/the-tail-at-scale/ | resilience.md | not checked |
| https://sre.google/sre-book/addressing-cascading-failures/ | resilience.md | checked 2026-11-16 |
| https://sre.google/sre-book/handling-overload/ | resilience.md | not checked |
| https://www.cosmicpython.com/book/chapter_02_repository.html | SKILL.md, architectural.md, structural.md | checked 2026-11-16 |
| https://www.cosmicpython.com/book/chapter_04_service_layer.html | SKILL.md, architectural.md | checked 2026-11-16 |
| https://www.cosmicpython.com/book/chapter_06_uow.html | architectural.md | checked 2026-11-16 |
| https://www.cosmicpython.com/book/chapter_08_events_and_message_bus.html | architectural.md | checked 2026-11-16 |
| https://www.cosmicpython.com/book/chapter_09_all_messagebus.html | architectural.md | not checked |
| https://www.cosmicpython.com/book/chapter_12_cqrs.html | architectural.md | checked 2026-11-16 |
| https://www.cosmicpython.com/book/chapter_13_dependency_injection.html | SKILL.md, creational.md | checked 2026-11-16 |
| https://www.cosmicpython.com/book/epilogue_1_how_to_get_there_from_here.html | architectural.md | not checked |
| https://www.destroyallsoftware.com/screencasts/catalog/functional-core-imperative-shell | more-patterns.md | not checked |
| https://www.destroyallsoftware.com/talks/boundaries | more-patterns.md | not checked |
| https://www.domainlanguage.com/wp-content/uploads/2016/05/DDD_Reference_2015-03.pdf | architectural.md | not checked |
