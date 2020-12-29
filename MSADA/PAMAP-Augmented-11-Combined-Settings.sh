
win_size=( 128 )
overlap=( 0.80 )
activity=11
source1=( 1 2 5 8)
source2=( 1 2 5 8)
target=( 1 2 5 8)

url=$PWD
file_url="/MSADA/PAMAP-SamePosition-DifferentUser-Augmented-11/PAMAP-SamePosition-DifferentUser-Augmented-11.txt"
echo "S1_User S1_Pos S2_User S2_Pos Tar_User Tar_Pos Target_Acc Tar_Precision Tar_F1 Tar_Recall" >> $url$file_url


END=2

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
                            python MSADA-PAMAP-SamePosition-DifferentUser-Augmented-11.py --win_size ${win} --overlap ${ovarlap} --source1 ${s1} --source2 ${s2} --target ${tar} --activity $activity --position ${i}
                        fi
                    done
                done
            done
        done
    done
done



win_size=(128)
window_overlap=(0.80)
activity=11

source1=(0 1 2)
source2=(0 1 2)
target=(0 1 2)
user=(1 2 5 8)

END=2

url=$PWD
file_url="/MSADA/PAMAP-SameUser-DifferentPosition-Augmented-11/PAMAP-SameUser-DifferentPosition-Augmented-11.txt"
echo "S1_User S1_Pos S2_User S2_Pos Tar_User Tar_Pos Tar_Acc Tar_Precision Tar_F1 Tar_Recall" >> $url$file_url

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
                            python MSADA-PAMAP-SameUser-DifferentPosition-Augmented-11.py --win_size ${var4}  --overlap ${var5}  --source1 ${source1[$var1]} --source2 ${source2[$var2]} --target ${target[$var3]} --activity $activity --user ${i}
                        done
                    done
                fi
            done
        done
    done
done