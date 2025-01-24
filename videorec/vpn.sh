#!/bin/bash

# OpenVPN ile kullanılabilecek TLS şifreleme algoritmalarını al
for cipher in $(openvpn --show-tls | grep -o 'TLS-[A-Za-z0-9_\-]*'); do
  echo "Testing with cipher: $cipher"

  # OpenVPN istemcisini arka planda başlat ve bağlantı denemesini 10 saniyede sınırla
  timeout 10s openvpn \
    --client \
    --remote 192.168.1.17 \
    --auth-user-pass login.conf \
    --dev tun \
    --ca ca.crt \
    --auth-nocache \
    --comp-lzo \
    --tls-cipher "$cipher" &>/dev/null

  # Bağlantının başarılı olup olmadığını kontrol et
  if [ $? -eq 0 ]; then
    echo "Connection successful with cipher: $cipher"
    break
  else
    echo "Failed with cipher: $cipher"
  fi
done
