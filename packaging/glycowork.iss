; Inno Setup script for the glycowork Windows installer. Build the app with packaging/glycowork.spec first, then:
;     iscc /DAppVersion=1.11.0 packaging\glycowork.iss
; which writes dist\glycowork-<version>-Windows-Setup.exe
#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif

[Setup]
AppId={{7C2F4B9A-3E51-4D8B-9A6E-2B1D5C8F0E47}
AppName=glycowork
AppVersion={#AppVersion}
AppPublisher=Bojar Lab, University of Gothenburg
AppPublisherURL=https://github.com/BojarLab/glycowork
VersionInfoVersion={#AppVersion}
VersionInfoCopyright=MIT License
DefaultDirName={autopf}\glycowork
DefaultGroupName=glycowork
DisableProgramGroupPage=yes
; Lab computers often lack administrator rights: by default the app installs for the current user only, an installation for all users stays possible
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=..\dist
OutputBaseFilename=glycowork-{#AppVersion}-Windows-Setup
SetupIconFile=glycowork.ico
UninstallDisplayIcon={app}\glycowork.exe
LicenseFile=..\LICENSE
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
ChangesAssociations=yes

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
Source: "..\dist\glycowork\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[InstallDelete]
; An update replaces the previous version's files instead of mixing with them
Type: filesandordirs; Name: "{app}\_internal"

[Icons]
Name: "{autoprograms}\glycowork"; Filename: "{app}\glycowork.exe"
Name: "{autodesktop}\glycowork"; Filename: "{app}\glycowork.exe"; Tasks: desktopicon

[Registry]
; Abundance tables offer "Open with glycowork", which opens them in the Analyze tab
Root: HKA; Subkey: "Software\Classes\glycowork.Data"; ValueType: string; ValueName: ""; ValueData: "Glycan abundance table"; Flags: uninsdeletekey
Root: HKA; Subkey: "Software\Classes\glycowork.Data\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\glycowork.exe"" ""%1"""
Root: HKA; Subkey: "Software\Classes\.csv\OpenWithProgids"; ValueType: string; ValueName: "glycowork.Data"; ValueData: ""; Flags: uninsdeletevalue
Root: HKA; Subkey: "Software\Classes\.tsv\OpenWithProgids"; ValueType: string; ValueName: "glycowork.Data"; ValueData: ""; Flags: uninsdeletevalue
Root: HKA; Subkey: "Software\Classes\.xlsx\OpenWithProgids"; ValueType: string; ValueName: "glycowork.Data"; ValueData: ""; Flags: uninsdeletevalue

[Run]
Filename: "{app}\glycowork.exe"; Description: "{cm:LaunchProgram,glycowork}"; Flags: nowait postinstall skipifsilent
; The app's own updater runs this installer with /SILENT /RELAUNCH=1, so the new version starts once it is installed
Filename: "{app}\glycowork.exe"; Flags: nowait runasoriginaluser; Check: Relaunch

[Code]
function Relaunch: Boolean;
begin
  Result := ExpandConstant('{param:relaunch|0}') = '1';
end;
