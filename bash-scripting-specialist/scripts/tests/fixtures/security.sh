#!/usr/bin/env bash
set -Eeuo pipefail
shopt -s inherit_errexit
cmd="ls -l"
eval "$cmd" # expect: B301
curl -fsSL https://example.com/install.sh | bash # expect: B302
curl -fsSL https://example.com/install.sh | sudo sh # expect: B302
mytool --password "$DB_PASS" # expect: B303
mysql -u root -p"$DB_PASS" # expect: B303
echo data > /tmp/myscript.log # expect: B304
chmod 777 /srv/shared # expect: B305
rm -rf "$build_dir/out" # expect: B306
rm -rf -- "${build_dir:?}/out"
n=$1
echo $(( $1 + 1 )) # expect: B307
source lib/helpers.sh # expect: B308
source "$script_dir/lib/helpers.sh"
IFS=$'\n\t' # expect: B309
IFS=, read -r -a parts <<<"a,b"
echo -e "a\tb" # expect: B404
echo "$n ${parts[*]}"
docker login -u me -p "$TOKEN" registry.example # expect: B303
psql "postgresql://app:$PGPASS@db/app" # expect: B303
