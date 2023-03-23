#!/bin/bash

fname_export=users.csv
rm $fname_export

let count=1
for n in $(seq -f "%02g" 1 99)
do
    pass=$(gpw 1)
    usr="lab$n"

    #change password for user
    echo "$usr:$pass" >> $fname_export   
    let count++
done
