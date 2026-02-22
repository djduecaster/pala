JETSON_HOST ?= jetson
JETSON_DIR ?= ~/pala
LOG_RUNS_ROOT ?= logs/runs
SSH_FLAGS ?= -T -o LogLevel=ERROR
RUN_ID ?=

.PHONY: deploy run go pull-logs

deploy:
	./deploy_jetson.sh

run:
	./run_jetson.sh

go:
	./deploy_jetson.sh && ./run_jetson.sh

pull-logs:
	@set -e; \
	if [ -n "$(RUN_ID)" ]; then \
		target_run="$(JETSON_DIR)/logs/runs/$(RUN_ID)"; \
	else \
		target_run=$$(ssh $(SSH_FLAGS) "$(JETSON_HOST)" "ls -1d $(JETSON_DIR)/logs/runs/[0-9]* 2>/dev/null | sort | tail -n 1" 2>/dev/null || true); \
	fi; \
	target_run=$$(printf '%s' "$$target_run" | tr -d '\r' | tr -d '\n'); \
	if [ -z "$$target_run" ]; then \
		echo "❌ No run-scoped logs found on Jetson at $(JETSON_DIR)/logs/runs"; \
		exit 1; \
	fi; \
	run_name=$$(basename "$$target_run"); \
	dst="$(LOG_RUNS_ROOT)/$$run_name"; \
	mkdir -p "$$dst"; \
	echo "ℹ️ Pulling run-scoped logs: $$run_name"; \
	rsync -az "$(JETSON_HOST):$$target_run/" "$$dst/"; \
	echo "✅ Pull complete: $$dst"
