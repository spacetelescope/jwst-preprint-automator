# CLAUDE.md
Your goal is to refactor the JWST Preprint DOI automator. 

## Workflow
- Do all work in a git branch, "dev-refactor".
- Work interactively with the user.
- Do not try to do *perfect* software design; just make it good enough and simple to understand.
- Eventually, iterate with the user on writing unit tests.
- Do *not* include Claude in commit messages, i.e., DO NOT WRITE `Co-Authored-By: Claude` anywhere

## Scope
- Focus on refactoring and enhancing the code in the `jwst_preprint_analyzer/` module.
- Current priority: Implement arXiv ID pagination to overcome ADS API 2000-row limit
- You may look at the README, and if needed, the prompt files. 
- Ignore all results, notebooks, and other data sources. 

## Refactoring Objectives

### 1. Module Restructuring
- **Current State**: Monolithic Python script
- **Target**: Well-organized multi-file module structure
- **Follow**: Python packaging best practices (PEP 8, separation of concerns)
- **Consider**: 
  - `__init__.py` for package structure
  - Separate modules for distinct functionality
  - Clear import hierarchy
  - Avoid circular dependencies

### 2. Code Cleanup
- **Remove**: Excessive comments that don't add value
- **Preserve**: Comments explaining non-trivial business logic
- **Focus**: Self-documenting code through clear naming and structure

### 3. Documentation Updates
- **Phase 1**: Reference existing README.md (well-written baseline)
- **Phase 2**: Update README.md to reflect new structure post-refactoring
- **Maintain**: API documentation and usage examples

## Key Principles
- **Single Responsibility**: Each module should have one clear purpose
- **DRY**: Eliminate code duplication across modules
- **Maintainability**: Code should be easy to understand and modify

## Current Implementation Task
Implement arXiv ID pagination to overcome ADS API 2000-row limit:
- Enhance existing `--year-month` mode with pagination based on arXiv ID patterns
- Break queries into chunks like `arXiv:YYMM.0*`, `arXiv:YYMM.1*`, etc. to stay under 2000-row limit
- ArXiv IDs don't reflect submission dates, so date-range queries via ADS API are not feasible
- ADS API only provides `pubdate` (publication date) not arXiv submission date
- Pagination combined with existing filtering will keep result sets manageable
- Maintain existing query semantics and report format
- No rate limiting concerns for human usage patterns
