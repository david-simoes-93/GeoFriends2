max="$1"
count=`expr $max - 1`

export PYTHONPATH=$(pwd)

echo 0 to "$count"
sleep 5

for i in `seq 0 $count`
do
    python3 DQN/DQN-Distributed.py --slaves_per_url="$max" --urls=localhost --task_index="$i" &
    echo python3 DQN/DQN-Distributed.py --task_max="$max" --task_index="$i"
    sleep 5
done
