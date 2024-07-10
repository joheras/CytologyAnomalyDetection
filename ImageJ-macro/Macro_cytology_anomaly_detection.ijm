/*
 * 
 * GOAL: using a anomaly detection model, classifying the images of cytology (patches of them) as anomalous
 *    
 * INPUT: path of a folder with the files (tif-format) where each file contains different series of images, which were adquired by different objectives
 * 			
 * OUTPUT: a folder with the exported images and a csv-file which contains the information about the patches of the images which have been classify as anomalous
 * 
 * Last Update July 2024
 */
 
//VARIABLES
serie=3; //the number of the serie to export the images
outputFolder="_Output_S"+serie; //name of the output folder
batchMode=true; //to enable the background process //it is true, the macro does not work.

//PROCESS 
print("\\Clear");
setForegroundColor(255, 255, 255);
run("Bio-Formats Macro Extensions");
setBatchMode(batchMode);

//choose the folder
#@ File (label="Choose folder with the experiments", style="directory") directory

dirParent = File.getParent(directory);
dirName = File.getName(directory);
dirOutput = dirParent+File.separator+dirName+outputFolder;
if (File.exists(dirOutput)==false) {
  	File.makeDirectory(dirOutput); // new output folder
}
dirOutImgS3=dirOutput+File.separator+"imgS3_rois";
if (File.exists(dirOutImgS3)==false) {
     File.makeDirectory(dirOutImgS3); // new output folder
}
//step 1: export the Serie 3 of the images
print("Starting...");
list = getFileList(directory);
print("number of files in the folder: "+list.length);
for (i=0; i<list.length; i++) //list.length
{
     pathI=directory+File.separator+list[i];
     print(pathI);
     //only tif-files
     indexExt= lastIndexOf(list[i], "."); //indexOf(string, substring) Returns the index within string of the last occurrence of substring
     extension= substring(list[i], indexExt);
     if(extension==".tif"){
          run("Bio-Formats", "open=pathI autoscale color_mode=Default view=Hyperstack stack_order=XYCZT series_"+serie);
          title=getTitle();
          auxTitle=split(title, "C");
          name=auxTitle[0];
          run("Stack to RGB");
          saveAs("Tiff", dirOutImgS3+File.separator+name+".tif"); //Image Serie 3
          run("Close All");;
     }
}

//step 2: obtaining the predictions

print("applying the model...");
timeO=getTime();

exec("sh", "-c", "cytology-anomaly-detection "+dirOutImgS3);
timeL=getTime();//Returns the current time in milliseconds.
time=(timeL-timeO)/1000;
print("total time: "+time+"sec");
print("Done!");


