# Official workflow adaptation tracking

The user has authorized adapting all official workflows, with H3 first.
The application frontend will call native vLLM APIs directly. ComfyUI integration
is excluded by the user's subsequent clarification; it is not an acceptance
dependency. Retain the generation capabilities shown by official examples.

The source contract is the H3 recipe at Omni
`7be014bce6374f06c95b703763bdbac4c6198f31`, also compared against
`031612f2507594ef632f923379e04ee36ff98684`. These two copies of the recipe are
identical. Source implementation/tests take precedence over contradictory old
prose about Ref2VA accepting only one image and one audio.

Status is tracked separately for implementation, CPU validation, real generation,
and output quality. A registered parameter or a fake-engine HTTP test is not a
completed workflow. Pending items below remain authorized work.

| Area | Implemented / evidence | Remaining acceptance or implementation |
| --- | --- | --- |
| T2VA | Native base path and real four-step Turbo short generation | Full duration, additional seeds, visual/audio review |
| First, last, first+last FL2VA | Native inputs; real first+last Turbo short generation | Independent first/last and full quality gates |
| Ref2VA media combinations | Native image/video/audio packing and input checks; API matrix | Complete real mixed-reference run and combination quality |
| Reference-video segment selection | Native start offsets, media validation and API mapping | Real segment-selection evidence |
| Remote API inputs | JSON, multipart, HTTP(S), data URLs implemented with CPU coverage | GPU generation through the actual public API |
| Video task lifecycle | Async/sync, polling/list/download/delete, multi-output, OpenAPI | Native engine integration and frontend-facing error validation |
| LightX2V four/eight-step family | Eight artifact contracts; two four-step FL2V real cases | Ref2V, eight-step, artifact and original-base coverage |
| FlashGen four-step T2VA | Native 259-target loader, DMD2 schedule and original AdaLN restoration; CPU tests | Real GPU generation, measured residency and quality, including restored INT8 base |
| FastH3 Dense four-step T2VA | Original-weight streaming fusion, all 343 targets, native TP loading and CPU staging; fixed four-step API | GPU generation, startup/residency and quality; separate INT8 integration |
| FastH3 VSA | Audited learned gates and sparse attention | SM70 execution strategy, numerical and quality validation |
| Dynamic adapter selection | One loaded adapter and request scale | Multi-adapter loading, eviction, stage/partition binding |
| Request quality / Cache-DiT | Lossless path only | High profile, refresh hints and quality comparison |
| TeaCache | Audited FL2VA-only restriction | Native FL2VA cache, uncached Ref2VA routing and quality |
| Combined FL2VA/Ref2VA service | One partition per engine currently | Shared encoder/VAEs, per-request DiT selection, memory budget |
| Encoder/diffusion stage split | Audited upstream two-stage and Turbo deploy | Native transport, stage ownership and matching outputs |
| Step execution / continuous batching | Request-serial engine currently | Step admission, packed attention isolation and measured benefit |
| CPU/DLO residency / parallel decode | Native pinned staging and tiled TP VAE | Official residency variants and deployment combinations |
| FP8 / SAGE / Skip-Softmax | Official hardware-specific recipes inventoried | SM70 feasibility or supported equivalent, explicit accuracy gates |
| Other model families | Official inventory in `../omni_workflow_coverage.md` | Native pipelines and their user workflows after H3 priority work |

The initial implementation order is public API and reference-media completion,
remaining distilled adapter families, then shared-stage/routing/cache/scheduler
work. Each API must describe only implemented settings. Hardware-specific paths
must demonstrate an actual supported kernel; forwarding an unsupported backend
name or silently ignoring a flag does not satisfy acceptance.

Source ownership stays on the task branch/worktree recorded in [CONTROL.md](CONTROL.md)
and PR #565. The user requested merging this source checkpoint to `main`;
the pending GPU/quality gates in this table remain open. Kernel performance work remains independently
owned. Do not preempt their GPU leases to produce workflow evidence.
