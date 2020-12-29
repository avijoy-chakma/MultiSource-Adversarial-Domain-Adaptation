
win_size=( 128 )
overlap=( 0.90 )
activity=10
source1=( 1 2 3 4 5 6 7 8)
source2=( 1 2 3 4 5 6 7 8)
target=( 1 2 3 4 5 6 7 8)


url=$PWD
file_url="/MSADA/MSADA-DSADS-SamePosition-DifferentUser-Augmented-10/MSADA-DSADS-SamePosition-DifferentUser-Augmented-10.txt"
echo "S1_User S1_Pos S2_User S2_Pos Tar_User Tar_Pos A_Acc A_Precision A_F1 A_Recall B_Acc B_Precision B_F1 B_Recall" >> $url$file_url


END=4

for i in $(seq 0 $END); 
do
    for win in "${win_size[@]}"
    do
        for ovarlap in "${overlap[@]}"
        do 
            for s1 in "${source1[@]}"
            do
                for s2 in "${source2[@]}"
                do
                    for tar in "${target[@]}"
                    do
                        if [ $s1 != $s2 -a $s2 != $tar -a $tar != $s1 -a $s2 -gt $s1 ]; then
                            echo "Position" $i "S1=" $s1 "S2=" $s2 "Tar=" $tar
                            python MSADA-DSADS-SamePosition-DifferentUser-Augmented-10.py --win_size ${win} --overlap ${ovarlap} --source1 ${s1} --source2 ${s2} --target ${tar} --activity $activity --position ${i}
                        fi
                    done
                done
            done
        done
    done
done