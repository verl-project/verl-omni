
- [ ] pre: add github govern, including pre-commit ci workflow

### Phase 1: flowgrpo runnable with minimal verl code
- basic code
- run check
Done

### Phase 2: runnable without verl folder, import from verl
- draft README
- vllm, vllm-omni version, verl version. installtation guide

- docs
    - add reference results

### Phase 3: add CI tests

- light-weight CI
    - cpu unit tests

- heavy CI
    - e2e smoke tests

### Phase 4: add AI co-work rules


### Future TODOs:
- simplified pre-commit-config.yaml，remove autogen-trainer-cfg hook
- installation setup: use pyproject.toml 

- data process script local_dir should be removed!! 

- version release, verl-omni v0.1

### Communicate
- 尽快跟外部开发者交代

-------------------------
Re-define the task

stage 0: code repo set up

stage 1: doc system 


---- 
My commits:
- add the flowgrpo algo figure
- update readme page
- Supported features or models

---- 
Critical concern from xibin:
- how verl and verl-omni called or imported or registered each other
    - custom pipelines
    - rollout backends / trainer backends 注入
- verl_omni的目录结构是否合理
    - experimental -> rollut?
- verl_omni的架构图，类似于vllm-omni给一个
    - design doc也可以挂在docs system, dev_guide/design_of_verl_omni
- API design
    ==> docs补一下api reference






