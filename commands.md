
`find . -type f -name '._*' -delete`
`find . -type f -name '.DS_Store' -delete`

SSH接続が切れても処理を続けるコマンド．永続ではない．
直後の1コマンドにのみ適用される．
`nohup python your_script.py > output.log 2>&1 &`


=====================
スパコン利用に関する手引き
=====================

https://genkai-portal.hpc.kyushu-u.ac.jp/user-portal/myspc

からログイン(ワンタイムパスワードは，Appleのパスワードアプリ)

https://www.cc.kyushu-u.ac.jp/scp/system/ITO/02_login/3-3.html

これに従いコマンドを叩けばいけるらしいが，タイムアウトする


コマンドのパスワードは　kosen

# スパコン接続コマンド

ssh -i /Users/murakamitakumi/Documents/research/spcom_keys/id_rsa -l ku50001550 genkai.hpc.kyushu-u.ac.jp

# スパコンのルートディレクトリ
/home/pj24003114/ku50001550/research-prj


# ファイル転送コマンド：ローカル→スパコン
scp -i /Users/murakamitakumi/Documents/research/spcom_keys/id_rsa \
    -r `/path/to/local/directory` \
    ku50001550@genkai.hpc.kyushu-u.ac.jp:`/path/to/remote/destination/`



scp -i /Users/murakamitakumi/Documents/research/spcom_keys/id_rsa \
    -r /Users/murakamitakumi/Desktop/make_10fold_dataset.py \
    ku50001550@genkai.hpc.kyushu-u.ac.jp:/home/pj24003114/ku50001550/research-prj/data/



# ファイル転送コマンド：スパコン→ローカル
scp -i /Users/murakamitakumi/Documents/research/spcom_keys/id_rsa \
    -r ku50001550@genkai.hpc.kyushu-u.ac.jp:`/path/to/remote/directory` \
    `/path/to/local/destination/`