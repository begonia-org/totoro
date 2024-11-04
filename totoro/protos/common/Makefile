# Makefile

# 定义变量
PROTOC = protoc
PYTHON = python3
TS_PROTO_PLUGIN = $(shell which protoc-gen-ts_proto)

# 目标文件夹
PROTO_DIR = .
OUTPUT_DIR = ../../
PY_OUTPUT_DIR = ../../../
# 文件列表
GO_PROTO_FILES = $(wildcard $(PROTO_DIR)/begonia/api/v1/*.proto)
PY_PROTO_FILES = $(wildcard $(PROTO_DIR)/begonia/api/v1/*.proto)
TS_PROTO_FILES = $(wildcard $(PROTO_DIR)/begonia/api/v1/*.proto)
COMMON_ARGS = options.proto common.proto
# 通用参数
GO_ARGS = --go_out=$(OUTPUT_DIR) --go_opt=paths=source_relative \
          --go-grpc_out=$(OUTPUT_DIR) --grpc-gateway_out=$(OUTPUT_DIR) \
          --grpc-gateway_opt=paths=source_relative --go-grpc_opt=paths=source_relative
PY_ARGS = -m grpc_tools.protoc -I=$(PROTO_DIR) --python_out=$(OUTPUT_DIR) \
          --pyi_out=$(PY_OUTPUT_DIR) --grpc_python_out=$(PY_OUTPUT_DIR) 
TS_ARGS = --plugin="protoc-gen-ts=$(TS_PROTO_PLUGIN)" \
          --ts_proto_opt=esModuleInterop=true --ts_proto_opt=paths=source_relative \
          --ts_proto_opt=snakeToCamel=false --ts_proto_opt=oneof=unions \
          --ts_proto_out=$(OUTPUT_DIR)


make_dir:
	@mkdir -p $(OUTPUT_DIR)

.PHONY: make_dir generate go python ts

generate: make_dir go python ts

# 生成 Go 代码
go: $(GO_PROTO_FILES) | make_dir
	$(PROTOC) -I=$(PROTO_DIR) $(GO_ARGS) $?
	protoc-go-inject-tag -input="$(OUTPUT_DIR)begonia/api/v1/*.pb.go"

# 生成 Python 代码
python: $(PY_PROTO_FILES) | make_dir
	touch $(PY_OUTPUT_DIR)__init__.py
	echo "#!/usr/bin/env python" > $(PY_OUTPUT_DIR)__init__.py
	$(PYTHON) $(PY_ARGS) $?
	sed -i '/from/!s/import \(.*\) as/from . import \1 as/g' $(OUTPUT_DIR)*.py*

# 生成 TypeScript 代码
ts: $(TS_PROTO_FILES) | make_dir
	$(PROTOC) -I=$(PROTO_DIR) $(TS_ARGS) $?


