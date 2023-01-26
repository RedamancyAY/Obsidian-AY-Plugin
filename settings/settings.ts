import MyPlugin from "../main";
import { App, PluginSettingTab, Setting } from "obsidian";
import { FolderSuggest } from "./suggesters/FolderSuggester";

export interface PluginSettings {
	dateFormat: string;
	pdf_folder: string;
	html_folder: string;
	img_folder: string;
}

export const DEFAULT_SETTINGS: Partial<PluginSettings> = {
	dateFormat: "YYYY-MM-DD",
	pdf_folder: "",
	html_folder: "",
	img_folder: "",
};


export class SettingTab extends PluginSettingTab {
    plugin: MyPlugin;

    constructor(app: App, plugin: MyPlugin) {
        super(app, plugin);
        this.plugin = plugin;
    }

    display(): void {
        let { containerEl } = this;

        containerEl.empty();

        containerEl.createEl("h1", { text: "AY's 插件" });

        containerEl.createEl("h2", { text: "测试插件面板" });
        new Setting(containerEl)
            .setName("Date format")
            .setDesc("Default date format")
            .addText((text) =>
                text
                    .setPlaceholder("MMMM dd, yyyy")
                    .setValue(this.plugin.settings.dateFormat)
                    .onChange(async (value) => {
                        this.plugin.settings.dateFormat = value;
                        await this.plugin.saveSettings();
                    })
            );

        this.add_file_default_folder_setting();
    }

    // 1. 为附件设置默认移动的文件夹
    add_file_default_folder_setting(): void {
        this.containerEl.createEl("h2", { text: "移动附件" });
        new Setting(this.containerEl)
            .setName("PDF files")
            .setDesc("Default folder")
            .addSearch((cb) => {
                new FolderSuggest(cb.inputEl);
                cb.setPlaceholder("Example: folder1/folder2")
                    .setValue(this.plugin.settings.pdf_folder)
                    .onChange((new_folder) => {
                        this.plugin.settings.pdf_folder = new_folder;
                        this.plugin.saveSettings();
                    });
                // @ts-ignore
                cb.containerEl.addClass("SettingTab_folder_search");
            });
        new Setting(this.containerEl)
            .setName("HTML files")
            .setDesc("Default folder")
            .addSearch((cb) => {
                new FolderSuggest(cb.inputEl);
                cb.setPlaceholder("Example: folder1/folder2")
                    .setValue(this.plugin.settings.html_folder)
                    .onChange((new_folder) => {
                        this.plugin.settings.html_folder = new_folder;
                        this.plugin.saveSettings();
                    });
                // @ts-ignore
                cb.containerEl.addClass("SettingTab_folder_search");
            });
        new Setting(this.containerEl)
            .setName("Image files")
            .setDesc("Default folder")
            .addSearch((cb) => {
                new FolderSuggest(cb.inputEl);
                cb.setPlaceholder("Example: folder1/folder2")
                    .setValue(this.plugin.settings.img_folder)
                    .onChange((new_folder) => {
                        this.plugin.settings.img_folder = new_folder;
                        this.plugin.saveSettings();
                    });
                // @ts-ignore
                cb.containerEl.addClass("SettingTab_folder_search");
            });
    }
}