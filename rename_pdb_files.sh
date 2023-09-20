cd [your pdb directory]
gunzip -r .
for file in */*; do mv $file ${file/pdb/}; done
for file in */*; do mv $file ${file/.ent/.pdb}; done
