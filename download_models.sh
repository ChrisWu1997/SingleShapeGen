mkdir -p checkpoints
cd checkpoints


if [ -z "$1" ]
  then
    ## all checkpoint tags
    declare -a arr=(
        "ssg_Acropolis_r128s6" "ssg_Acropolis_r256s8" "ssg_BoulderStone_r256s8" 
        "ssg_Cactus_r128s6" "ssg_Canyon_r256s8" "ssg_Castle_r256s8" "ssg_Cheese_r128s6" 
        "ssg_CubeHotel_r256s8" "ssg_CurvedVase_r192s8" "ssg_DesertCactus_r256s9" 
        "ssg_ElmTree_r256s9" "ssg_FloatingWood_r256s8" "ssg_IcelandicMountain_r256s9" 
        "ssg_IndustryHouse_r256s8" "ssg_Log_r256s8" "ssg_NaturalArch_r256s9" 
        "ssg_PlantPot_r256s8" "ssg_RockStairs_r256s8" "ssg_Rock_r128s7" 
        "ssg_RuinedBuilding_r128s6" "ssg_SmallTown_r256s8" "ssg_Stalagmites_r128s7" 
        "ssg_Stalagmites_r192s8" "ssg_StoneStairs_r128s7" "ssg_StoneWall_r256s8" 
        "ssg_Table_r256s8" "ssg_Terrain2_r256s9" "ssg_Terrain_r128s6" 
        "ssg_Terrain_r256s8" "ssg_Tree_r128s6" "ssg_Vase_r128s7" 
        "ssg_Vase_r192s8" "ssg_Wall_r128s6" "ssg_WaterMountain_r256s9"
        "singan3d_Acropolis_r128s6" "singan3d_Cactus_r128s6" "singan3d_Cheese_r128s6" 
        "singan3d_Rock_r128s7" "singan3d_Stalagmites_r128s7" "singan3d_StoneStairs_r128s7" 
        "singan3d_Terrain_r128s6" "singan3d_Tree_r128s6" "singan3d_Vase_r128s7" "singan3d_Wall_r128s6" 
    )
else
    declare -a arr=("$1")
fi


for tag in "${arr[@]}"
do
    echo $tag
    link="http://www.cs.columbia.edu/cg/SingleShapeGen/pretrained/$tag.tar.gz"
    wget $link
    if [ -f $tag.tar.gz ]; then
        tar -xvzf $tag.tar.gz
        rm $tag.tar.gz
    else
        echo "$tag.tar.gz not found"
    fi
done
