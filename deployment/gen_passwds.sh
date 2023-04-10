#!/bin/bash

fname_export=users.csv
rm $fname_export

for n in $(seq -f "%02g" 0 99)
do
    pass=$(pwgen 10 1)
    usr="lab$n"

    #change password for user
    echo "$usr:$pass" >> $fname_export   
done
