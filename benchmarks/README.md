# benchmarks/

Output folder for evaluation runs and full-coverage build previews. Contents are generated and not tracked.
The `tools` Compose service mounts this folder at `/app/benchmarks`; keep it so Docker does not create it as root.
