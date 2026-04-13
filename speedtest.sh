#!/usr/bin/env bash

best_provider=""
best_latency=999999

for p in $(cc-switch --app codex provider list \
  | grep '^│' \
  | grep -v 'ID' \
  | awk -F '┆' '{gsub(/ /, "", $2); if ($2 != "" && $2 != "ID") print $2}'); do
  
  echo "==== Testing $p ===="
  
  output=$(cc-switch --app codex provider speedtest "$p" 2>&1)
  echo "$output"
  
  latency=$(printf '%s\n' "$output" | awk -F '┆' '
    /^│/ && $2 ~ /[0-9][[:space:]]*ms/ {
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", $2)
      gsub(/[[:space:]]*ms$/, "", $2)
      print $2
      exit
    }
  ')
  
  if [ -n "$latency" ]; then
    echo "Parsed latency: ${latency} ms"
    
    if [ "$latency" -lt "$best_latency" ]; then
      best_latency=$latency
      best_provider=$p
    fi
  else
    echo "Could not parse latency for $p"
  fi
  
  echo
done

if [ -n "$best_provider" ]; then
  echo "Switching to fastest provider: $best_provider (${best_latency} ms)"
  cc-switch --app codex provider switch "$best_provider"
else
  echo "No valid latency found."
fi