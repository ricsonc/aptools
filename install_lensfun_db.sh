#!/usr/bin/env bash

wget https://github.com/lensfun/lensfun/archive/refs/tags/v0.3.95.tar.gz
tar -xf v0.3.95.tar.gz
mv lensfun-0.3.95/data/db lensfun_db
rm v0.3.95.tar.gz
rm -r lensfun-0.3.95
