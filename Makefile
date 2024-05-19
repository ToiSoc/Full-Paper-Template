# 定义变量

TEX_FILE := TJJM-GoicO.tex
BIB_FILE := TJJM-GoicO.bib
AUX_FILE := $(TEX_FILE:.tex=.aux)
OUTPUT_DIR := Tmp
PDF_OUTPUT := $(OUTPUT_DIR)/$(basename $(TEX_FILE)).pdf


# 创建输出目录（如果不存在）
$(OUTPUT_DIR):
	mkdir -p $@


# XeLaTeX 编译循环并将PDF放入输出目录
.PHONY: xe
x: $(OUTPUT_DIR)
	xelatex -output-directory=$(OUTPUT_DIR) $(TEX_FILE)
	biber $(OUTPUT_DIR)/$(basename $(TEX_FILE)); \
	xelatex -output-directory=$(OUTPUT_DIR) $(TEX_FILE); \
	xelatex -output-directory=$(OUTPUT_DIR) $(TEX_FILE); \


# pdfLaTeX 编译循环并将PDF放入输出目录
.PHONY: p
p: $(OUTPUT_DIR)
	pdflatex -output-directory=$(OUTPUT_DIR) $(TEX_FILE)
	@if grep -q 'Rerun to get (cross-references|citation info)' $(OUTPUT_DIR)/$(basename $(TEX_FILE)).aux; then \
		biber $(OUTPUT_DIR)/$(basename $(TEX_FILE)) || bibtex $(OUTPUT_DIR)/$(basename $(TEX_FILE)); \
		pdflatex -output-directory=$(OUTPUT_DIR) $(TEX_FILE); \
		pdflatex -output-directory=$(OUTPUT_DIR) $(TEX_FILE); \
	fi


# 单独运行XeLaTeX并将PDF放入输出目录
.PHONY: xo
xo: $(OUTPUT_DIR)
	xelatex -output-directory=$(OUTPUT_DIR) $(TEX_FILE)


# 默认目标设置为x
.DEFAULT_GOAL := x


# 清理辅助文件及输出目录中的PDF文件
.PHONY: clean
clean:
	rm -rf $(OUTPUT_DIR)


# 仅清理输出目录中的PDF文件
.PHONY: clean-pdf
clean-pdf:
	rm -f $(PDF_OUTPUT)
