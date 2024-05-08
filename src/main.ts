import { Editor, MarkdownView, Notice, Plugin, TFile, TAbstractFile, Menu, WorkspaceLeaf, TFolder } from 'obsidian';
import { SettingTab, PluginSettings, DEFAULT_SETTINGS } from "./settings/settings";
import { debugLog, path } from './utils/utils';
import { isImage, isVideo, isAudio } from './utils/check_attachments';
import { getAPI } from "obsidian-dataview";

const dv_api = getAPI();

export default class MyPlugin extends Plugin {
	settings: PluginSettings;

	async onload() {
		await this.loadSettings();


		var vault_name = this.app.vault.adapter.getName();
		var _vault_path = this.app.vault.adapter;
		if (_vault_path != null && 'basePath' in _vault_path){
			var vault_path = _vault_path.basePath;
		}
		debugLog('hello', vault_name, vault_path);


		// This creates an icon in the left ribbon.
		const ribbonIconEl = this.addRibbonIcon('dice', 'Sample Plugin', (evt: MouseEvent) => {
			// Called when the user clicks the icon.
			new Notice('Hello, this is AY');
		});
		// Perform additional things with the ribbon
		ribbonIconEl.addClass('my-plugin-ribbon-class');

		// This adds a status bar item to the bottom of the app. Does not work on mobile apps.
		const statusBarItemEl = this.addStatusBarItem();
		statusBarItemEl.setText('Status Bar Text');

		// 0-设置面板
		this.addSettingTab(new SettingTab(this.app, this));
		this.add_folder_sep();
		// this.registerInterval(
		// 	window.setInterval(() => this.add_folder_sep(), 1000)
		//   );

		this.registerEvent(
			this.app.workspace.on('file-menu', (menu: Menu, file: TAbstractFile, source: string, leaf?: WorkspaceLeaf) => {
				console.log("hello, world!!!")
				this.add_folder_sep();
				console.log("hello, world!!!");
			})
		)


		// 1. set color for selected text
		this.addCommand({
			id: 'text-background-pink',
			name: '改变文字背景颜色为➡️粉色',
			editorCallback: (editor: Editor, view: MarkdownView) => {
				const selection = editor.getSelection();
				const pre_str = '<span class="text-bg-pink">'
				const back_str = '</span>'
				if (selection.length > 0) {
					editor.replaceSelection(pre_str + selection + back_str);
				}
				else {
					new Notice('You didn\'t select any text!');
				}
			}
		});
		this.addCommand({
			id: 'text-background-green',
			name: '改变文字背景颜色为➡️绿色🟩',
			editorCallback: (editor: Editor, view: MarkdownView) => {
				const selection = editor.getSelection();
				const pre_str = '<span class="text-bg-green">'
				const back_str = '</span>'
				if (selection.length > 0) {
					editor.replaceSelection(pre_str + selection + back_str);
				}
				else {
					new Notice('You didn\'t select any text!');
				}
			}
		});
		this.addCommand({
			id: 'text-background-purple',
			name: '改变文字背景颜色为➡️紫色🟪',
			editorCallback: (editor: Editor, view: MarkdownView) => {
				const selection = editor.getSelection();
				const pre_str = '<span class="text-bg-purple">'
				const back_str = '</span>'
				if (selection.length > 0) {
					editor.replaceSelection(pre_str + selection + back_str);
				}
				else {
					new Notice('You didn\'t select any text!');
				}
			}
		});

		this.addCommand({
			id: 'move file into folder',
			name: '移动文件到特定文件夹',
			checkCallback: (checking: boolean) => {
				const markdownView = this.app.workspace.getActiveViewOfType(MarkdownView);
				if (markdownView) {
					if (!checking) {
						const filename = markdownView.file?.name;
						const filepath = markdownView.file?.path;
						debugLog('filename', filepath, filename);
						const metadata = getAPI(this.app)?.page(filepath);
						let dst_path = `Area/${metadata.Area}/${metadata.sub_area}`;
						if (metadata.subsub_area) {
							dst_path += `/${metadata.subsub_area}`;
							debugLog(metadata.subsub_area);
						}
						debugLog(metadata, dst_path, 'hello', vault_path);
						if (markdownView.file != null){
							this.move_files(markdownView.file, dst_path);
						}
					}
					return true;
				}
			},
		});

		this.addCommand({
			id: 'move file into other vault',
			name: '移动文件到其他库',
			checkCallback: (checking: boolean) => {
				const markdownView = this.app.workspace.getActiveViewOfType(MarkdownView);
				if (markdownView) {
					if (!checking) {
						const fs = require('fs');
						const filename = markdownView.file?.name;
						const filepath = vault_path + "/" + markdownView.file?.path;
						debugLog('filename', filepath, filename);

						var other_vault_path = this.settings.other_vault_path;
						debugLog("valut path is: ", other_vault_path, filepath, fs.existsSync(filepath), fs.existsSync(other_vault_path));
						
						fs.copyFileSync(filepath, other_vault_path + "/" + filename);

					}
					return true;
				}
			},
		});



		this.registerEvent(
			this.app.vault.on('create', (file) => {
				if (!(file instanceof TFile))
					return

				const timeGapMs = Date.now() - file.stat.ctime;
				// if the pasted file is created more than 1 second ago, ignore it
				if (timeGapMs > 1000)
					return

				if (isImage(file)) {
					debugLog('pasted image: ', file)
					this.move_files(file, this.settings.img_folder);
				}
				else if (file.extension.toLowerCase() == "html") {
					debugLog('pasted html: ', file)
					this.move_files(file, this.settings.html_folder);
				}
				else if (file.extension.toLowerCase() == "pdf") {
					debugLog('pasted PDF: ', file)
					this.move_files(file, this.settings.pdf_folder);
				}
				else if (isAudio(file)) {
					debugLog('pasted audio: ', file)
					this.move_files(file, this.settings.audio_folder);
				}
				else if (isVideo(file)) {
					debugLog('pasted video: ', file)
					this.move_files(file, this.settings.video_folder);
				}
			})
		)

		// When registering intervals, this function will automatically clear the interval when the plugin is disabled.
		this.registerInterval(window.setInterval(() => console.log('setInterval'), 5 * 60 * 1000));
	}

	onunload() {

	}

	async loadSettings() {
		this.settings = Object.assign({}, DEFAULT_SETTINGS, await this.loadData());
	}
	async saveSettings() {
		await this.saveData(this.settings);
	}

	// handleFileMenu(menu, file) {
	// 	console.log(file);
	// 	if (file instanceof TFolder) {
	// 		// Perform actions when a folder is clicked
	// 		console.log("Folder clicked:", file.name);
	// 	}
	// }

	async move_files(file: TFile, folder: string) {
		const new_path = path.join(folder, file.name)
		debugLog(file.name, file.parent?.path, new_path);

		//// check the folder exists 
		const folder_cls = this.app.vault.getAbstractFileByPath(folder);
		if (!folder_cls) {
			try {
				await this.app.vault.createFolder(folder);
			}
			catch (err) {
				new Notice(`❌Failed to create ${folder}: ${err}`)
				throw err
			}
		}

		//// move file
		try {
			await this.app.vault.rename(file, new_path);
		}
		catch (err) {
			new Notice(`❌Failed to move ${this.settings.img_folder}: ${err}`)
			throw err
		}
		new Notice(`✅Success!!! Move ${file.name} to ${new_path}`)
	}
	async add_folder_sep() {
		// 1. 首先删除所有的分割线
		let elements = document.querySelectorAll("[class^=folder-separator-]");
		for (let i = 0; i < elements.length; i++) {
			elements[i].remove();
		}

		// 2. 从设置中，读取要添加分割线的文件夹名
		const folders_sep_before = this.settings.folder_sep_before.split(',');
		const folders_sep_after = this.settings.folder_sep_after.split(',');

		// 3. 给文件夹加分割线
		const fileExplorer = document.querySelector(".nav-folder-children");
		if (fileExplorer) {
			const folderListItems = fileExplorer.querySelectorAll(".nav-folder-children > .nav-folder");
			for (let i = 0; i <= folderListItems.length - 1; i++) {
				// 3.1 获取当前的文件夹名
				const cur_folder = folderListItems[i].querySelector(".nav-folder-title");
				if (!cur_folder) continue;
				const cur_folder_name = cur_folder.getAttribute("data-path");
				if (!cur_folder_name) continue;
				// console.log(cur_folder_name);

				// 3.2 判断当前文件夹是否在设置里，是的话，就添加分割线
				if (folders_sep_before.includes(cur_folder_name)) {
					const hrElement = folderListItems[i].querySelector(".folder-separator-before");
					if (!hrElement) {
						const newHrElement = document.createElement("hr");
						newHrElement.classList.add("folder-separator-before");
						folderListItems[i].parentNode?.insertBefore(newHrElement, folderListItems[i]);
					}
				}
				if (folders_sep_after.includes(cur_folder_name)) {
					const hrElement = folderListItems[i].querySelector(".folder-separator-after");
					if (!hrElement) {
						const newHrElement = document.createElement("hr");
						newHrElement.classList.add("folder-separator-after");
						// folderListItems[i].append(newHrElement);
						folderListItems[i].parentNode?.insertAfter(newHrElement, folderListItems[i]);
						// newHrElement.insertAfter(folderListItems[i]?.parentElement?.parentElement);
					}
				}
			}
		}
	}
}


