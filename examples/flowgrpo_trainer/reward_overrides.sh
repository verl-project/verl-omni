#!/usr/bin/env bash

# Switch the launcher's defaults when the caller uses a legacy Hydra key. Keep
# caller arguments intact so Hydra still interprets their operators and values.
apply_legacy_reward_overrides() {
    local override key index
    for override in "$@"; do
        key=${override%%=*}
        key=${key#++}
        key=${key#+}
        key=${key#"~"}
        case "$key" in
            reward.reward_functions|reward.reward_functions.*)
                for index in "${!reward_scorer_args[@]}"; do
                    override=${reward_scorer_args[index]}
                    key=${override%%=*}
                    key=${key/+reward.scorers./+reward.reward_functions.}
                    key=${key/.kwargs./.}
                    reward_scorer_args[index]="${key}=${override#*=}"
                done
                reward_scorer_args+=(
                    reward.custom_reward_function.path=pkg://verl_omni.reward_loop.reward_manager.multi
                    reward.custom_reward_function.name=_multi_reward_placeholder
                )
                return
                ;;
        esac
    done
}
