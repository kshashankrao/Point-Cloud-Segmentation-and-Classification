for file in *.off;
	do meshlabserver -i $(basename $file .off).off -o $(basename $file .off).ply; 
done

