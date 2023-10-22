import { TFile, TAbstractFile } from 'obsidian';


function isContainExt(file: TAbstractFile, EXTS: Array<string>): boolean{
	if (file instanceof TFile) {
		if (EXTS.contains(file.extension.toLowerCase())) {
			return true
		}
	}
	return false
}


// check whether the file is image files 
export function isImage(file: TAbstractFile): boolean {
	// exts that denote the file is image
	const _EXTS = [
		'jpg', 'jpeg', 'png', 'gif', 'tiff', 'tif'
	]	
	return isContainExt(file, _EXTS)
}

// check whether the file is video files 
export function isVideo(file: TAbstractFile): boolean {
	// exts that denote the file is video
	const _EXTS = [
		'mp4', 'avi', 'mov', 'flv', 'mkv'
	]
	return isContainExt(file, _EXTS)
}

// check whether the file is audio files 
export function isAudio(file: TAbstractFile): boolean {
	// exts that denote the file is audio
	const _EXTS = [
		'mp3', 'wav', 'flac', 'aac'
	]
	return isContainExt(file, _EXTS)
}