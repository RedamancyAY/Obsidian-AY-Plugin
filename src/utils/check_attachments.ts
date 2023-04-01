import { TFile, TAbstractFile } from 'obsidian';


export function isImage(file: TAbstractFile): boolean {
	const _EXTS = [
		'jpg', 'jpeg', 'png', 'gif', 'tiff', 'tif'
	]	
	if (file instanceof TFile) {
		if (_EXTS.contains(file.extension.toLowerCase())) {
			return true
		}
	}
	return false
}

export function isVideo(file: TAbstractFile): boolean {
	const _EXTS = [
		'mp4', 'avi', 'mov', 'flv', 'mkv'
	]
	if (file instanceof TFile) {
		if (_EXTS.contains(file.extension.toLowerCase())) {
			return true
		}
	}
	return false
}

export function isAudio(file: TAbstractFile): boolean {
	const _EXTS = [
		'mp3', 'wav', 'flac', 'aac'
	]
	if (file instanceof TFile) {
		if (_EXTS.contains(file.extension.toLowerCase())) {
			return true
		}
	}
	return false
}