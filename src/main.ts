import { Editor, MarkdownView, Notice, Plugin, TFile, TAbstractFile } from 'obsidian';
import { SettingTab, PluginSettings, DEFAULT_SETTINGS } from "./settings/settings";
import { debugLog, path, isImage } from './utils/utils';


export default class MyPlugin extends Plugin {
	settings: PluginSettings;

	async onload() {
		await this.loadSettings();

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

		// If the plugin hooks up any global DOM events (on parts of the app that doesn't belong to this plugin)
		// Using this function will automatically remove the event listener when this plugin is disabled.
		// this.registerDomEvent(document, 'click', (evt: MouseEvent) => {
		// 	console.log('click', evt);
		// });


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

	async move_files(file: TFile, folder: string) {
		const new_path = path.join(folder, file.name)
		console.log(file.name, file.parent.path, new_path);
		try {
			await this.app.vault.rename(file, new_path);
		}
		catch (err) {
			new Notice(`Failed to move ${this.settings.img_folder}: ${err}`)
			throw err
		}
		new Notice(`Move ${file.name} to ${new_path}`)
	}
}


