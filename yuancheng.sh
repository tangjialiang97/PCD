rsync -avz --progress \
  --filter=':- .gitignore' \
  -e "ssh -p 26996" \
  ./ root@192.10.84.169:/opt/data/private/tjl/code/TSLF


rsync -avz --progress -e "ssh -p 26996" ./data/ root@192.10.84.169:/opt/data/private/tjl/code/TSLF/data/
