import { Plugin, TFile, Vault } from 'obsidian';

  
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
