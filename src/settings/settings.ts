import MyPlugin from "../main";
import { App, PluginSettingTab, Setting } from "obsidian";
import { FolderSuggest } from "./suggesters/FolderSuggester";
import { debugLog } from "src/utils/utils";


// variables for plugin settings
export interface PluginSettings {
    dateFormat: string;
    pdf_folder: string;
    html_folder: string;
    img_folder: string;
    audio_folder: string;
    video_folder: string;
    folder_sep_before: string;
    folder_sep_after: string;
    bg_img_folder: string; // the folder that store background images
    bg_img_left: string; // name of the left bg image
    bg_img_left_width: string; // width of the left bf image
    bg_img_right: string; // name of the right bg image
    bg_img_right_width: string; // width of the right bf image
    other_vault_name: string;
    other_vault_path: string;
}

export const DEFAULT_SETTINGS: Partial<PluginSettings> = {
    dateFormat: "YYYY-MM-DD",
    pdf_folder: "",
    html_folder: "",
    img_folder: "",
    audio_folder: "",
    video_folder: "",
    folder_sep_before: "",
    folder_sep_after: "",
    bg_img_folder: "",
    bg_img_left: "",
    bg_img_left_width: "100px",
    bg_img_right: "",
    bg_img_right_width: "100px",
    other_vault_name: "",
    other_vault_path: "",
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
        this.add_folder_sep_setting();
        this.add_bg_img_setting();
        this.add_move_vault_setting();
    }

    // 1. 为附件设置默认移动的文件夹
    add_file_default_folder_setting(): void {
        this.containerEl.createEl("h2", { text: "移动附件" });
        new Setting(this.containerEl)
            .setName("PDF files")
            .setDesc("[.pdf]")
            .addSearch((cb) => {
                new FolderSuggest(this.app, cb.inputEl);
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
            .setDesc("[.html]")
            .addSearch((cb) => {
                new FolderSuggest(this.app, cb.inputEl);
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
            .setDesc("[.jpg, .png, .tiff]")
            .addSearch((cb) => {
                new FolderSuggest(this.app, cb.inputEl);
                cb.setPlaceholder("Example: folder1/folder2")
                    .setValue(this.plugin.settings.img_folder)
                    .onChange((new_folder) => {
                        this.plugin.settings.img_folder = new_folder;
                        this.plugin.saveSettings();
                    });
                // @ts-ignore
                cb.containerEl.addClass("SettingTab_folder_search");
            });
        new Setting(this.containerEl)
            .setName("Audio files")
            .setDesc("[.wav, .mp3]")
            .addSearch((cb) => {
                new FolderSuggest(this.app, cb.inputEl);
                cb.setPlaceholder("Example: folder1/folder2")
                    .setValue(this.plugin.settings.audio_folder)
                    .onChange((new_folder) => {
                        this.plugin.settings.audio_folder = new_folder;
                        this.plugin.saveSettings();
                    });
                // @ts-ignore
                cb.containerEl.addClass("SettingTab_folder_search");
            });
        new Setting(this.containerEl)
            .setName("Video files")
            .setDesc("[.mp4, .avi]")
            .addSearch((cb) => {
                new FolderSuggest(this.app, cb.inputEl);
                cb.setPlaceholder("Example: folder1/folder2")
                    .setValue(this.plugin.settings.video_folder)
                    .onChange((new_folder) => {
                        this.plugin.settings.video_folder = new_folder;
                        this.plugin.saveSettings();
                    });
                // @ts-ignore
                cb.containerEl.addClass("SettingTab_folder_search");
            });
    }


    // 2. 给文件夹添加分割线
    add_folder_sep_setting(): void {
        this.containerEl.createEl("h2", { text: "设置文件夹分隔符" });
        new Setting(this.containerEl)
            .setName("UP")
            .setDesc("添加在文件夹上面")
            .addSearch((cb) => {
                new FolderSuggest(this.app, cb.inputEl);
                cb.setPlaceholder("")
                    .setValue(this.plugin.settings.folder_sep_before)
                    .onChange((new_folder) => {
                        this.plugin.settings.folder_sep_before = new_folder;
                        this.plugin.saveSettings();
                        this.plugin.add_folder_sep();
                    });
                // @ts-ignore
                cb.containerEl.addClass("SettingTab_folder_search");
            });
        new Setting(this.containerEl)
            .setName("Down")
            .setDesc("添加在文件夹上面")
            .addSearch((cb) => {
                new FolderSuggest(this.app, cb.inputEl);
                cb.setPlaceholder("")
                    .setValue(this.plugin.settings.folder_sep_after)
                    .onChange((new_folder) => {
                        this.plugin.settings.folder_sep_after = new_folder;
                        this.plugin.saveSettings();
                        this.plugin.add_folder_sep();
                    });
                // @ts-ignore
                cb.containerEl.addClass("SettingTab_folder_search");
            });
    }

    // 3. 添加背景图像文件夹
    add_bg_img_setting():void{
        this.containerEl.createEl("h2", { text: "添加背景图像" });
        new Setting(this.containerEl)
            .setName("左侧图像名")
            .setDesc("文件名")
            .addText((text) =>
                text
                    .setPlaceholder("/path/of/imgs")
                    .setValue(this.plugin.settings.bg_img_left)
                    .onChange(async (value) => {
                        this.plugin.settings.bg_img_left = value;
                        this.changeLeftBgImg(value);
                        await this.plugin.saveSettings();
                    })
            )
            .addText((text) =>
                text
                    .setPlaceholder("image width, 100px")
                    .setValue(this.plugin.settings.bg_img_left_width)
                    .onChange(async (value) => {
                        this.plugin.settings.bg_img_left_width = value;
                        this.changeLeftBgImgWidth(value);
                        await this.plugin.saveSettings();
                    })
            );
        new Setting(this.containerEl)
            .setName("右侧侧图像名")
            .setDesc("文件名")
            .addText((text) =>
                text
                    .setPlaceholder("/path/of/imgs")
                    .setValue(this.plugin.settings.bg_img_right)
                    .onChange(async (value) => {
                        this.plugin.settings.bg_img_right = value;
                        this.changeRightBgImg(value);
                        await this.plugin.saveSettings();
                    })
            )
            .addText((text) =>
                text
                    .setPlaceholder("image width, 100px")
                    .setValue(this.plugin.settings.bg_img_right_width)
                    .onChange(async (value) => {
                        this.plugin.settings.bg_img_right_width = value;
                        this.changeRightBgImgWidth(value);
                        await this.plugin.saveSettings();
                    })
            );
        
    }

    changeLeftBgImgWidth(width: string) {  
        debugLog('change left bg img size');
        const root = document.documentElement; 
        root.style.setProperty('--nav-files-container-bg_img-size', width);  
    }  
    changeLeftBgImg(url: string) {  
        debugLog('change left bg img');
        const root = document.documentElement; 
        root.style.setProperty('--nav-files-container-bg_img', 'url('+ url+')');  
    }  
    changeRightBgImgWidth(width: string) {  
        debugLog('change left bg img size');
        const root = document.documentElement; 
        root.style.setProperty('--note-bg_img-size', width);  
    } 
    changeRightBgImg(url: string) {  
        debugLog('change right bg img');
        const root = document.documentElement; 
        root.style.setProperty('--note-bg_img', 'url('+ url+')');  
    }  

    // 4. 
    add_move_vault_setting():void{
        this.containerEl.createEl("h2", { text: "其他valut路径" });
        new Setting(this.containerEl)
            .setName("vault名和路径")
            .setDesc("必须是系统的绝对路径")
            .addText((text) =>
                text
                    .setPlaceholder("Vault Name")
                    .setValue(this.plugin.settings.other_vault_name)
                    .onChange(async (value) => {
                        this.plugin.settings.other_vault_name = value;
                        await this.plugin.saveSettings();
                    })
            )
            .addText((text) =>
                text
                    .setPlaceholder("Vault Path")
                    .setValue(this.plugin.settings.other_vault_path)
                    .onChange(async (value) => {
                        this.plugin.settings.other_vault_path = value;
                        await this.plugin.saveSettings();
                    })
            );
        
    }
}