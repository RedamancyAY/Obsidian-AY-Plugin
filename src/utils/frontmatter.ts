import { Plugin, TFile, Vault } from 'obsidian';
import { debugLog, path } from './utils';
import { getAPI } from "obsidian-dataview";


export async function clearFrontmatter(file: TFile|null, vault: Vault) {
    if (file == null){
        console.error('Error clearing frontmatter: null input');
        return true;
    }
    
    try  {
        const fileData = await vault.read(file); // Read file content
        const modifiedContent = removeFrontmatter(fileData); // Remove frontmatter
        await vault.modify(file, modifiedContent); // Save modified content back to file
    } catch (error) {
        console.error('Error clearing frontmatter:', error);
    }
}

function removeFrontmatter(content: string): string {
    // Regex pattern to match frontmatter (supports both --- and ```yaml styles)
    const frontmatterRegex = /^---\s*\n(?:.|\n)*?\n---\s*\n*/;

    // Remove frontmatter from content
    const modifiedContent = content.replace(frontmatterRegex, '');

    return modifiedContent.trim(); // Trim to remove leading/trailing whitespace
}

export async function create_tag_from_Area(file: TFile | null){
    if (file == null){
        return true;
    }
    const filepath = file.path;
    const metadata = getAPI(this.app)?.page(filepath);
    let tag = "";
    if (metadata.Area){
        tag += `Area/${metadata.Area}`;
    }
    if (metadata.sub_area){
        tag += `/${metadata.sub_area}`;
    }
    if (metadata.subsub_area) {
        tag += `/${metadata.subsub_area}`;
    }
    let metadata_tags = metadata.tags;
    if (tag == "") return true;
    if (file){
        this.app.fileManager.processFrontMatter(file, (fm: { tags: string[] | undefined; }) =>{
            if (fm.tags == undefined){
                fm.tags = [];
                debugLog("Create tags item in the metadata of file: ", filepath);
            }
            const index = fm.tags?.indexOf(tag);
            if (index == undefined || index != -1) return true;
            else{
                fm.tags.splice(0, 0, tag);
                debugLog("Create tag ", tag, "in the file: ", filepath);
            }
        })
    }
}
export async function create_Area_from_tag(file: TFile | null){
    if (file == null){
        return true;
    }
    const filepath = file.path;
    const metadata = getAPI(this.app)?.page(filepath);
    let tags = metadata.tags;
    for (let i =0 ; i < tags.length; i++){
        if (tags[i].indexOf("Area/") != -1){
            if (file){
                this.app.fileManager.processFrontMatter(file, (fm: { Area: any; sub_area: any; subsub_area: any; }) =>{
                    let sub_tags = tags[i].split('/')
                    for (let j = 1; j < sub_tags.length; j++){
                        if (j == 1) fm.Area = sub_tags[j];
                        if (j == 2) fm.sub_area = sub_tags[j];
                        if (j == 3) fm.subsub_area = sub_tags[j];
                        if (j > 3) break;
                    }
                })
            }		
            break;
        }
    }
}