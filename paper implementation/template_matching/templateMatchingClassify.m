fid = fopen('results.txt','w');
files1 = dir('test_files\*.jpg');
for file1 = files1'
    test_image = (strcat('test_files\',file1.name));

    english_count = 0;
    hindi_count = 0;
    telugu_count = 0;
    %iterate over files
    disp('iterating english templates');
    files = dir('templates\english_templates\*.jpg');
    for file = files'
        template = strcat('templates\english_templates\',(file.name));
        n = templateMatching(template,test_image);
        english_count = english_count + n;
    end

    disp('iterating hindi templates');
    files = dir('templates\hindi_templates\*.jpg');
    for file = files'
        template = strcat('templates\hindi_templates\',(file.name));
        n = templateMatching(template,test_image);
        hindi_count = hindi_count + n;
    end

    disp('iterating telugu templates');
    files = dir('templates\telugu_templates\*.jpg');
    for file = files'
        template = strcat('templates\telugu_templates\',(file.name));
        n = templateMatching(template,test_image);
        telugu_count = telugu_count + n;
    end
    
    final_string  = strcat(test_image,strcat(' , ',strcat(num2str(english_count),strcat(' , ',strcat(num2str(hindi_count),strcat(' , ',num2str(telugu_count)))))));
    fprintf(fid, '%s ', final_string);
    fprintf(fid, '\n');
    % english_count
    % hindi_count
    % telugu_count
end

fclose(fid);