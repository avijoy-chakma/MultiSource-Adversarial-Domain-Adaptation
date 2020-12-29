win_size=(128)
window_overlap=(0.90)
activity=10

source1=(0 1 2 3 4)
source2=(0 1 2 3 4)
target=(0 1 2 3 4)
user=(1 2 3 4 5 6 7 8)


url=$PWD
file_url="/MSADA/MSADA-DSADS-SameUser-DifferentPosition-Augmented-10/MSADA-DSADS-SameUser-DifferentPosition-Augmented-10.txt"
echo "S1_User S1_Pos S2_User S2_Pos Tar_User Tar_Pos A_Acc A_Precision A_F1 A_Recall B_Acc B_Precision B_F1 B_Recall" >> $url$file_url

for i in "${user[@]}"
do 
    for var1 in "${source1[@]}"
    do
        for var2 in "${source2[@]}"
        do
            for var3 in "${target[@]}"
            do
                if [ $var1 != $var2 -a $var2 != $var3 -a $var1 != $var3 -a $var2 -gt $var1 ]; then
                    echo "Executing User ${i} ${source1[$var1]} ${source2[$var2]} ${target[$var3]}"; 
                    for var4 in "${win_size[@]}"
                    do
                        for var5 in "${window_overlap[@]}"
                        do 
                            python MSADA-DSADS-SameUser-DifferentPosition-Augmented-10.py --win_size ${var4}  --overlap ${var5}  --source1 ${source1[$var1]} --source2 ${source2[$var2]} --target ${target[$var3]} --activity $activity --user ${i}
                        done
                    done
                fi
            done
        done
    done
done