# Makefile

# 定义变量
PROTOC = protoc
PYTHON = python3
TS_PROTO_PLUGIN = $(shell which protoc-gen-ts_proto)

# TS_PROTO_PLUGIN = $(shell which protoc-gen-ts_proto)

# 目标文件夹
PROTO_DIR = .
OUTPUT_DIR = ../pb
MODELS_DIR= ../models
COMMON_DIR= ../common
# 生成的文件
TS_OUTPUT_DIR="../api"
# 文件列表
PY_PROTO_FILES = $(wildcard $(PROTO_DIR)/*.proto)
TS_PROTO_FILES = $(wildcard $(PROTO_DIR)/*.proto)

PY_ARGS = --python_out=$(OUTPUT_DIR) \
          --pyi_out=$(OUTPUT_DIR) --grpc_python_out=$(OUTPUT_DIR) --pydantic_out=$(MODELS_DIR)
TS_ARGS = --plugin="protoc-gen-ts=$(TS_PROTO_PLUGIN)" \
          --ts_proto_opt=esModuleInterop=true --ts_proto_opt=paths=source_relative \
          --ts_proto_opt=snakeToCamel=false --ts_proto_opt=oneof=unions \
		  --ts_proto_opt=Mgoogle/protobuf/field_mask.proto=begonia-sdk \
		  --ts_proto_opt=Mgoogle/protobuf/timestamp.proto=begonia-sdk \
		  --ts_proto_opt=Mgoogle/protobuf/any.proto=begonia-sdk \
		  --ts_proto_opt=Mgoogle/protobuf/descriptor.proto=begonia-sdk \
		  --ts_proto_opt=Mgoogle/protobuf/struct.proto=begonia-sdk \
		  --ts_proto_opt=Mgoogle/protobuf/empty.proto=begonia-sdk \
		  --ts_proto_opt=Mgoogle/api/annotations.proto=begonia-sdk \
		  --ts_proto_opt=Mgoogle/api/http.proto=begonia-sdk \
		  --ts_proto_opt=Mgoogle/api/httpbody.proto=begonia-sdk \
          --ts_proto_out=$(TS_OUTPUT_DIR)


common:
	@mkdir -p $(OUTPUT_DIR)
	@mkdir -p $(COMMON_DIR)

models:
	@mkdir -p $(MODELS_DIR)
.PHONY: make_dir generate python,common
clean:
	@rm -rf $(COMMON_PROTO_FILES)
# 生成所有代码
generate: make_dir py.print,common,desc


py: $(PY_PROTO_FILES) | common
	@mkdir -p $(MODELS_DIR)
	@mkdir -p $(OUTPUT_DIR)
	touch $(OUTPUT_DIR)/__init__.py
	echo "#!/usr/bin/env python" > $(OUTPUT_DIR)/__init__.py
	touch $(MODELS_DIR)/__init__.py
	echo "#!/usr/bin/env python" > $(MODELS_DIR)/__init__.py
	python3 -m grpc_tools.protoc --proto_path=$(PROTO_DIR) --proto_path=./common  $(PY_ARGS) $?
	sed -i '/from/!s/import \(.*\) as/from . import \1 as/g' $(OUTPUT_DIR)/*pb2*.py

ts: $(TS_PROTO_FILES) | common
	@mkdir -p $(TS_OUTPUT_DIR)
	$(PROTOC) -I=$(PROTO_DIR) -I=./common $(TS_ARGS) $?

desc:
	protoc --descriptor_set_out="../pb/totoro.bin" --include_imports -I=./ -I=./common ./common/begonia/api/v1/web.proto $(PY_PROTO_FILES)
print:
	@echo $(PY_PROTO_FILES)