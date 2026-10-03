# Anthropic Messages cache usage

When the server reports prompt cache statistics, `/v1/messages` maps
`prompt_tokens_details.cached_tokens` to `usage.cache_read_input_tokens`.
Enable prompt token details with `--enable-prompt-tokens-details` to include
the underlying statistics.

The mapping applies to non-streaming responses and to the `message_start`
and final `message_delta` events in streaming responses. An available count
of zero is reported as zero. Missing statistics are not fabricated.

`input_tokens` and `output_tokens` keep their existing values; this change
does not subtract cache hits from `input_tokens` or report cache creation
statistics.
