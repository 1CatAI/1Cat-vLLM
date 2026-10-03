# Flash-V100 in precompiled SM70 wheels

Precompiled builds can reuse a wheel containing the vLLM CUDA extensions but
missing the standalone Flash-V100 extensions. Such a build currently succeeds
and later fails when the SM70 attention backend captures CUDA graphs.

For explicitly requested Volta targets, a missing Flash-V100 bundle should use
the ordinary source build when its sources and CUDA compiler are available.
Otherwise the build should fail with an actionable message. Existing complete
bundles continue to be reused. Other architecture targets retain their current
behavior.

Validation covers incomplete and complete bundles, source-build prerequisites,
wheel contents and native imports, followed by a CUDA graph route check.
