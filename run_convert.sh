#!/bin/bash

script_path="${1:-convert.py}"
shift || true

debug=0
debug_convert=0
filtered_args=()
for arg in "$@"; do
    if [[ "$arg" == "--debug" ]]; then
        debug=1
    elif [[ "$arg" == "--debug_convert" ]]; then
        debug_convert=1
    else
        filtered_args+=("$arg")
    fi
done
set -- "${filtered_args[@]}"

script_base="$(basename "$script_path")"

if [[ "$script_base" == "convert_endpose.py" ]]; then
    instruction_spec="$1"
    if [[ -z "$instruction_spec" ]]; then
        echo "用法: $0 convert_endpose.py \"instruction\" [extra_args...]"
        echo "或: $0 convert_endpose.py \"instr1:l1,instr2:l2\" --range START END [extra_args...]"
        echo "可选: --debug --debug_convert"
        exit 1
    fi
    shift || true

    start_idx=""
    end_idx=""
    if [[ "${1:-}" == "--range" ]]; then
        start_idx="$2"
        end_idx="$3"
        if [[ -z "$start_idx" || -z "$end_idx" ]]; then
            echo "用法: $0 convert_endpose.py \"instruction\" --range START END [extra_args...]"
            exit 1
        fi
        shift 3 || true
    fi

    mapfile -t episode_entries < <(
        find . -maxdepth 1 -type d -name "episode*" \
        | while read -r dir; do
            base="$(basename "$dir")"
            num="${base#episode}"
            if [[ "$num" =~ ^[0-9]+$ ]]; then
                printf "%s\t%s\n" "$num" "$dir"
            else
                printf "999999\t%s\n" "$dir"
            fi
        done \
        | sort -n -k1,1 \
        | awk -F'\t' '{print $1"\t"$2}'
    )
    if [[ "${#episode_entries[@]}" -eq 0 ]]; then
        echo "未找到任何 episode 目录，退出"
        exit 1
    fi
    if [[ "$debug" -eq 1 ]]; then
        echo "按 episode 数字排序后的目录列表:"
        for entry in "${episode_entries[@]}"; do
            num="${entry%%$'\t'*}"
            dir="${entry#*$'\t'}"
            echo "  episode${num}: ${dir}"
        done
    fi

    hdf5_paths=()
    for entry in "${episode_entries[@]}"; do
        num="${entry%%$'\t'*}"
        dir="${entry#*$'\t'}"
        if [[ -n "$start_idx" && ( "$num" -lt "$start_idx" || "$num" -gt "$end_idx" ) ]]; then
            continue
        fi
        hdf5_file="$dir/data.hdf5"
        if [[ -f "$hdf5_file" ]]; then
            hdf5_paths+=("$hdf5_file")
        else
            echo "警告: 未找到 $hdf5_file"
        fi
    done
    if [[ -n "$start_idx" && "$debug" -eq 1 ]]; then
        echo "range 选择: [episode${start_idx}, episode${end_idx}] 共 ${#hdf5_paths[@]} 个"
        for i in "${!hdf5_paths[@]}"; do
            echo "  选中[$i] ${hdf5_paths[$i]}"
        done
    fi

    if [[ "${#hdf5_paths[@]}" -eq 0 ]]; then
        echo "未找到任何 data.hdf5，退出"
        exit 1
    fi

    if [[ "$instruction_spec" == *:* ]]; then
        IFS=',' read -r -a instr_pairs <<< "$instruction_spec"
        total_expected=0
        instr_list=()
        len_list=()
        instruction_lines=()
        for pair in "${instr_pairs[@]}"; do
            instr="${pair%:*}"
            len="${pair##*:}"
            if [[ -z "$instr" || -z "$len" || ! "$len" =~ ^[0-9]+$ ]]; then
                echo "指令格式错误: $pair"
                exit 1
            fi
            instr_list+=("$instr")
            len_list+=("$len")
            total_expected=$((total_expected + len))
        done

        if [[ "$total_expected" -ne "${#hdf5_paths[@]}" ]]; then
            echo "指令长度总和 ($total_expected) 不等于选中的 episode 数量 (${#hdf5_paths[@]})"
            exit 1
        fi

        for idx in "${!instr_list[@]}"; do
            instr="${instr_list[$idx]}"
            len="${len_list[$idx]}"
            for ((i=0; i<len; i++)); do
                instruction_lines+=("$instr")
            done
        done

        instr_file="$(mktemp ./instructions.XXXXXX.txt 2>/dev/null)" || instr_file="instructions_$$.txt"
        printf "%s\n" "${instruction_lines[@]}" > "$instr_file"

        if [[ "$debug" -eq 1 ]]; then
            for hdf5 in "${hdf5_paths[@]}"; do
                echo "将处理 hdf5: $hdf5"
            done
        fi
        echo "执行 python $script_path --hdf5_paths (共 ${#hdf5_paths[@]} 个) --instruction_file \"$instr_file\""
        if [[ "$debug" -eq 1 ]]; then
            extra_debug_flag=""
            if [[ "$debug_convert" -eq 1 ]]; then
                extra_debug_flag=" --debug"
            fi
            echo "debug 命令: python \"$script_path\" --hdf5_paths ${hdf5_paths[*]} --instruction_file \"$instr_file\"${extra_debug_flag} $*"
        fi

        if [[ "$debug_convert" -eq 1 ]]; then
            python "$script_path" --hdf5_paths "${hdf5_paths[@]}" --instruction_file "$instr_file" --debug "$@"
        else
            python "$script_path" --hdf5_paths "${hdf5_paths[@]}" --instruction_file "$instr_file" "$@"
        fi

        if [[ "$debug" -eq 1 ]]; then
            echo "instruction_file 路径: $instr_file"
        else
            rm -f "$instr_file"
        fi
    else
        if [[ "$debug" -eq 1 ]]; then
            for hdf5 in "${hdf5_paths[@]}"; do
                echo "将处理 hdf5: $hdf5"
            done
        fi
        echo "执行 python $script_path --hdf5_paths (共 ${#hdf5_paths[@]} 个) --instruction"
        if [[ "$debug" -eq 1 ]]; then
            extra_debug_flag=""
            if [[ "$debug_convert" -eq 1 ]]; then
                extra_debug_flag=" --debug"
            fi
            echo "debug 命令: python \"$script_path\" --hdf5_paths ${hdf5_paths[*]} --instruction \"$instruction_spec\"${extra_debug_flag} $*"
            if [[ "$debug_convert" -eq 1 ]]; then
                python "$script_path" --hdf5_paths "${hdf5_paths[@]}" --instruction "$instruction_spec" --debug "$@"
            else
                python "$script_path" --hdf5_paths "${hdf5_paths[@]}" --instruction "$instruction_spec" "$@"
            fi
        else
            python "$script_path" --hdf5_paths "${hdf5_paths[@]}" --instruction "$instruction_spec" "$@"
        fi
    fi
else
    # 遍历当前目录（.）下所有以 "episode" 开头的子文件夹
    # -maxdepth 1 确保只查找当前目录下的子文件夹，不深入嵌套
    # -type d 确保只选择目录（文件夹）
    find . -maxdepth 1 -type d -name "episode*" | while read dir; do
        # 进入找到的子文件夹
        cd "$dir" || { echo "无法进入目录 $dir"; continue; }

        # 在子文件夹中执行 python ../convert.py
        # ../ 会引用到上一级目录，即包含这个脚本的目录
        hdf5_file="data.hdf5"
        if [[ "$debug" -eq 1 ]]; then
            echo "将处理 hdf5: $dir/$hdf5_file"
        fi
        echo "进入 $dir 并执行 python ../$script_path"
        python "../$script_path" "$@"

        # 执行完毕后返回到原始目录，以便继续查找下一个文件夹
        cd ..
    done

    echo "所有操作已完成。"
fi
