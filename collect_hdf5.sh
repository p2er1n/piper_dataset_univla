#!/bin/bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: collect_hdf5.sh --out OUTPUT_DIR [--src ROOT] [--episode-start N] [--name-map MAP] [--tar [NAME]]

Collect converted .hdf5 files into a single directory and optionally create a tar.gz.

Options:
  --out, -o   Output directory (required)
  --src, -s   Root to search (default: .)
  --episode-start, -e  Start collecting from episode N (default: 0)
  --name-map, -n  Rename by map: name1+count1,name2+count2,...
  --tar, -t   Create tar.gz; optional NAME for archive (default: basename of output dir)
  --help, -h  Show this help
USAGE
}

src="."
out_dir=""
do_tar=0
archive_name=""
episode_start=0
name_map=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -s|--src)
      src="$2"
      shift 2
      ;;
    -o|--out)
      out_dir="$2"
      shift 2
      ;;
    -t|--tar)
      do_tar=1
      if [[ ${2-} != "" && ${2:0:1} != "-" ]]; then
        archive_name="$2"
        shift 2
      else
        shift
      fi
      ;;
    -e|--episode-start)
      episode_start="$2"
      shift 2
      ;;
    -n|--name-map)
      name_map="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$out_dir" ]]; then
  echo "Missing --out"
  usage
  exit 1
fi

if [[ "$src" != "/" ]]; then
  src="${src%/}"
fi

mkdir -p "$out_dir"

mapfile -d '' episode_dirs < <(find "$src" -maxdepth 1 -type d -name "episode*" -print0 | sort -z)

if [[ ${#episode_dirs[@]} -eq 0 ]]; then
  echo "No episode* directories found under $src"
  exit 1
fi

selected_dirs=()
for dir in "${episode_dirs[@]}"; do
  base="$(basename "$dir")"
  if [[ "$base" =~ ([0-9]+) ]]; then
    ep_num="${BASH_REMATCH[1]}"
    if (( ep_num < episode_start )); then
      continue
    fi
  fi
  selected_dirs+=("$dir")
done

if [[ ${#selected_dirs[@]} -eq 0 ]]; then
  echo "No episode* directories matched --episode-start $episode_start"
  exit 1
fi

files=()
for dir in "${selected_dirs[@]}"; do
  file="$dir/data_converted.hdf5"
  if [[ ! -f "$file" ]]; then
    echo "Missing data_converted.hdf5 in $dir"
    exit 1
  fi
  files+=("$file")
done

if [[ -n "$name_map" ]]; then
  IFS=',' read -r -a map_entries <<< "$name_map"
  map_total=0
  for entry in "${map_entries[@]}"; do
    if [[ ! "$entry" =~ ^([^+]+)\+([0-9]+)$ ]]; then
      echo "Invalid --name-map entry: $entry"
      exit 1
    fi
    map_total=$((map_total + BASH_REMATCH[2]))
  done
  if [[ $map_total -ne ${#files[@]} ]]; then
    echo "Name-map counts ($map_total) do not match episodes (${#files[@]})"
    exit 1
  fi

  file_idx=0
  for entry in "${map_entries[@]}"; do
    [[ "$entry" =~ ^([^+]+)\+([0-9]+)$ ]]
    name="${BASH_REMATCH[1]}"
    count="${BASH_REMATCH[2]}"
    for ((i=1; i<=count; i++)); do
      target="$(printf "%s+%02d.hdf5" "$name" "$i")"
      cp "${files[$file_idx]}" "$out_dir/$target"
      file_idx=$((file_idx + 1))
    done
  done
else
  for file in "${files[@]}"; do
    rel="${file#"$src"/}"
    flat="${rel//\//__}"
    cp "$file" "$out_dir/$flat"
  done
fi

if [[ $do_tar -eq 1 ]]; then
  if [[ -z "$archive_name" ]]; then
    archive_name="$(basename "$out_dir")"
  fi
  tar -czf "${archive_name}.tar.gz" -C "$out_dir" .
  echo "Created ${archive_name}.tar.gz"
fi

echo "Done."
