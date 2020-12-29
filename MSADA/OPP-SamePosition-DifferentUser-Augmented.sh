
win_size=( 128 )
overlap=( 0.80 )
activity=4
source1=( 1 2 3 4)
source2=( 1 2 3 4)
target=( 1 2 3 4)


url=$PWD
file_url="/MSADA/OPP-SamePosition-DifferentUser-Augmented/OPP-SamePosition-DifferentUser-Augmented.txt"
echo "S1_User S1_Pos S2_User S2_Pos Tar_User Tar_Pos Tar_Acc Tar_Precision Tar_F1 Tar_Recall" >> $url$file_url


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
                            python MSADA-OPP-SamePosition-DifferentUser-Augmented.py --win_size ${win} --overlap ${ovarlap} --source1 ${s1} --source2 ${s2} --target ${tar} --activity $activity --position ${i}
                        fi
                    done
                done
            done
        done
    done
done