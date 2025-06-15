# Directories containing submodules (excluding hidden directories starting with a dot)
SUBMODULE_DIRS := $(shell find . -mindepth 1 -maxdepth 1 -type d -not -path './.*')
# Variable that contains all submodules in the repository
SUBMODULES := $(shell git submodule --quiet foreach 'echo $$sm_path')
SHELL := /bin/bash

# Formatting codes
CYAN := \033[36m
GREEN := \033[1;32m
RESET := \033[0m
BOLD := \033[1m

# Common messages
MSG_COMPLETE := $(GREEN)Complete.$(RESET)

## General Commands

.PHONY: help
help:			
	@echo "==================================="
	@echo "  Django Project Makefile Commands "
	@echo "==================================="
	@awk -F':|##' ' \
		/^## / { \
			heading=substr($$0,4); \
			printf "\n\033[1;32m%s\033[0m\n", heading; \
		} \
		/^[a-zA-Z0-9_.-]+:.*##/ { \
			sub(/^[ \t]+/, "", $$1); \
			sub(/[ \t]+$$/, "", $$1); \
			sub(/^[ \t]+/, "", $$3); \
			sub(/[ \t]+$$/, "", $$3); \
			printf "  \033[36m%-20s\033[0m %s\n", $$1, $$3; \
		} \
	' $(MAKEFILE_LIST)


## GitHub CLI Commands

iauth: ## Login to GitHub CLI
	@gh auth login

istatus: ## Check the status of the GitHub repository
	@gh repo view --json name,description,defaultBranchRef

ils: ## List all open GitHub issues
	@gh issue list --state open

ilsa: ## List all GitHub issues (open + closed)
	@gh issue list --state all

ilsc: ## List all closed GitHub issues
	@gh issue list --state closed

icreate: ## Create a GitHub issue via interactive prompt, assigned to you, with 'enhancement' label
	@gh issue create --assignee "@me" --label "enhancement"; \
	gh issue list --state open

iedit: ils ## Edit a GitHub issue by number
	@read -p "Enter issue number to edit: " issue; \
	gh issue edit $$issue; \
	gh issue list --state all

idone: ils ## Close a GitHub issue by number
	@read -p "Enter issue number to close: " issue; \
	gh issue close $$issue --comment "Closed via Makefile"; \
	gh issue list --state all

ireopen: ilsc ## Reopen a GitHub issue by number
	@read -p "Enter issue number to reopen: " issue; \
	gh issue reopen $$issue --comment "Reopened via Makefile"; \
	gh issue list --state all

isub: ## Link a sub-issue (child) to a parent issue
	@read -p "Enter parent issue number: " parent; \
	read -p "Enter child issue number: " child; \
	gh issue develop $$parent --depends-on $$child; \
	echo "Linked #$$child as a sub-issue of #$$parent."
