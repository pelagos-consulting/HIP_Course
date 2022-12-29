#!/bin/bash

fname=users.csv
fname_export=users_export.csv
rm $fname

for n in $(seq -f "%02g" 1 99)
do
    pass=$(gpw 1)
    usr="lab$n"

    # Check if user exists
    if id -u "$usr" > /dev/null 1>&1
    then
        # User exists
        echo "$usr exists"
    else
        # Create the user and add to groups
        adduser $usr --disabled-password --gecos "" 
    fi

    # Check if jupyter-user exists
    if id -u "jupyter-$usr" > /dev/null 1>&1
    then
        # User exists
        echo "jupyter-$usr exists"
    else
        # Create the user and add to groups
        adduser jupyter-$usr --disabled-password --gecos "" 
    fi

    adduser $usr video
    adduser $usr render
    
    adduser jupyter-$usr video
    adduser jupyter-$usr render

    #change password for user
    echo "$usr:$pass" >> $fname
    echo "$usr:$pass" >> $export_fname    
    echo "jupyter-$usr:$pass" >> $fname
    
done

# Batch change user passwords
chpasswd < $fname
